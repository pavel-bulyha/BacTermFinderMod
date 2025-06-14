import os
import pickle
import re
import sys
import time
from datetime import datetime
from io import StringIO
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation, CompoundLocation
from Bio.SeqRecord import SeqRecord
from keras.models import load_model
from tqdm import tqdm
import shutil

file_format = 'genbank' #ATTENTION! The code is poorly optimized for the FASTA format - when using it, failures and inaccuracies are possible

def terminator_filter_decorator(func):
    """
    Decorator that, after performing the main processing (merging intervals),
    loads the original annotation from a GenBank file and filters the resulting intervals
    according to the following conditions:
      1. If an annotated terminator interval is completely contained within any original annotation interval.
      2. If an annotated terminator interval overlaps any original annotation interval by more than 25% of its length.
      3. If the annotated terminator interval overlaps any original annotation interval by more than 50 nucleotides.
      4. If there are annotations present, the distance from the end of the nearest original feature
         on the same strand to the start of the terminator must be ≤ 200 nucleotides; otherwise it is filtered out.
         If no annotations are found in the GenBank file, this distance filter is skipped and a warning
         is printed: "Attention: No annotation found in file, filtering by genes, ORFs and operons is not possible".

    After filtering, all intervals that are at most 75 nucleotides apart are merged together.
    The overall probability is calculated using the median, resulting in new start and end coordinates.
    Finally, if any interval exceeds 150 nucleotides in length, it is trimmed from both ends
    (by 5 nucleotides per iteration) until its length is less than or equal to 150
    (if already shorter, the trimming stops).

    The function extracts `genome_file` from **kwargs automatically. If `genome_file` is missing,
    an exception is raised. This allows post-processing to be optional and easily disabled.

    The function returns a pandas DataFrame with the refined intervals.

    Expected input DataFrame format (CSV columns separated by semicolon):

    | chrom       | strand | start | end  | probability_binary | probability_ENAC | probability_PS2 | probability_NCP | probability_mean |
    |-------------|--------|-------|------|--------------------|------------------|-----------------|-----------------|------------------|

    This decorator uses the 'strand' column of the input and compares it with the
    'strand' extracted from the GenBank file (converted to '+' or '-' accordingly).
    """
    def wrapper(*args, **kwargs):
        # Extract genome_file from kwargs
        genome_file = kwargs.get('genome_file')
        if genome_file is None:
            raise ValueError("Parameter 'genome_file' is required in **kwargs.")

        # Execute the main function to obtain the merged intervals DataFrame
        merged_df = func(*args, **kwargs)
        # merged_df must have the following columns:
        # 'chrom', 'strand', 'start', 'end', 'probability_binary',
        # 'probability_ENAC', 'probability_PS2', 'probability_NCP', 'probability_mean'

        # Load the GenBank file (assumes one record) and extract original annotation intervals
        record = SeqIO.read(genome_file, "genbank")
        original_intervals = []
        unique_intervals = set()  # A set for storing unique tuples of intervals
        for feature in record.features:
            # Exclude all records whose type (in lowercase) is 'source'
            if feature.type.lower() == "source":
                continue
            if not hasattr(feature, 'location'):
                continue

            # If the record has a complex arrangement (join), we process each part separately
            if isinstance(feature.location, CompoundLocation):
                parts = feature.location.parts
            else:
                parts = [feature.location]

            for part in parts:
                o_start = int(part.start)
                o_end = int(part.end)
                # Use a common strand for the feature (assuming it is the same for all parts)
                o_strand = '+' if feature.strand == 1 else '-' if feature.strand == -1 else '.'
                key = (record.id, o_start, o_end, o_strand)
                if key in unique_intervals:
                    continue
                unique_intervals.add(key)
                original_intervals.append({
                    'chrom': record.id,
                    'start': o_start,
                    'end': o_end,
                    'strand': o_strand
                })

        # Check if annotation exists
        has_annotation = len(original_intervals) > 0
        if not has_annotation:
            print("Attention: No annotation found in file, filtering by genes, ORFs and operons is not possible")

        # Helper function to calculate overlap length between two intervals
        def get_overlap(a_start, a_end, b_start, b_end):
            return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)

        # Filtering function: determines whether a candidate interval should be removed
        def should_remove(row):
            for orig in original_intervals:
                # Check if same chrom
                if row['chrom'] != orig['chrom']:
                    continue
                # Compare strands: only compare if original interval has a defined strand
                if orig['strand'] != '.' and row['strand'] != orig['strand']:
                    continue
                # Condition 1: candidate is completely contained within an original interval
                if orig['start'] <= row['start'] and orig['end'] >= row['end']:
                    return True
                # Calculate the overlapping nucleotides
                overlap = get_overlap(row['start'], row['end'], orig['start'], orig['end'])
                orig_length = orig['end'] - orig['start'] + 1
                # Condition 2: overlap is more than 25% of the original interval’s length
                # Condition 3: or overlap is more than 50 nucleotides
                if overlap > 0.25 * orig_length or overlap > 50:
                    return True

            # Condition 4: if annotations exist, distance to nearest feature must be ≤ 200 nt
            if has_annotation:
                if row['strand'] == '+':
                    # upstream for +: features whose END ≤ start of terminator
                    dists = [
                        row['start'] - orig['end']
                        for orig in original_intervals
                        if row['chrom'] == orig['chrom']
                        and orig['strand'] in ('.', '+')
                        and orig['end'] <= row['start']
                    ]
                elif row['strand'] == '-':
                    # upstream for –: features whose START ≥ end terminator
                    dists = [
                        orig['start'] - row['end']
                        for orig in original_intervals
                        if row['chrom'] == orig['chrom']
                        and orig['strand'] in ('.', '-')
                        and orig['start'] >= row['end']
                    ]
                else:
                    dists = []

                # if there is no upstream feature or the closest one is further than 200 nt - discard
                if not dists or min(dists) > 250:
                    return True

            return False

        # Remove intervals that must be filtered out
        filtered_df = merged_df[~merged_df.apply(should_remove, axis=1)].reset_index(drop=True)

        # Merge intervals with a gap of at most 75 nucleotides within the same 'chrom' and 'strand'
        prob_cols = ['probability_binary', 'probability_ENAC', 'probability_PS2',
                     'probability_NCP', 'probability_mean']

        def merge_intervals_gap(group: pd.DataFrame, gap: int = 75) -> pd.DataFrame:
            group = group.sort_values(by='start').reset_index(drop=True)
            merged = []
            current_start = group.iloc[0]['start']
            current_end = group.iloc[0]['end']
            current_probs = {col: [group.iloc[0][col]] for col in prob_cols}

            for i in range(1, len(group)):
                row = group.iloc[i]
                if row['start'] <= current_end + gap:
                    current_end = max(current_end, row['end'])
                    for col in prob_cols:
                        current_probs[col].append(row[col])
                else:
                    aggregated = {col: pd.Series(current_probs[col]).median() for col in prob_cols}
                    merged.append({
                        'chrom': group.iloc[0]['chrom'],
                        'strand': group.iloc[0]['strand'],
                        'start': current_start,
                        'end': current_end,
                        **aggregated
                    })
                    current_start = row['start']
                    current_end = row['end']
                    current_probs = {col: [row[col]] for col in prob_cols}

            aggregated = {col: pd.Series(current_probs[col]).median() for col in prob_cols}
            merged.append({
                'chrom': group.iloc[0]['chrom'],
                'strand': group.iloc[0]['strand'],
                'start': current_start,
                'end': current_end,
                **aggregated
            })
            return pd.DataFrame(merged)

        merged_groups = []
        for (ch, strand), group in filtered_df.groupby(['chrom', 'strand']):
            merged_group = merge_intervals_gap(group, gap=75)
            merged_groups.append(merged_group)
        merged_filtered_df = pd.concat(merged_groups, ignore_index=True)

        # Trim intervals longer than 150 nucleotides: repeatedly remove 5 nt from both ends until <= 150 nt.
        def trim_interval(row):
            while (row['end'] - row['start'] + 1) > 150:
                row['start'] += 5
                row['end'] -= 5
                if row['end'] < row['start']:
                    break
            return row

        trimmed_df = merged_filtered_df.apply(trim_interval, axis=1)
        return trimmed_df.reset_index(drop=True)

    return wrapper


def extract_sliding_windows(ref_genome_file: str, window_size: int,
                            step_size: int) -> pd.DataFrame:
    """This function extracts sliding windows from a reference genome.

    Args:
        ref_genome_file (str): this is the path to the reference genome file
        window_size (int): this is the size of the sliding window
        step_size (int): this is the step size of the sliding window

    Returns:
        pd.DataFrame: this is a Pandas DataFrame containing the sliding windows
    """
    # Load reference genome into memory
    ref_genome = SeqIO.to_dict(SeqIO.parse(ref_genome_file, file_format))

    # Define function to extract windows from a single sequence
    def extract_windows_from_sequence(seq_id):
        seq = ref_genome[seq_id].seq
        seq_len = len(seq)
        windows = []
        u_id = 0
        for i in tqdm(range(0, seq_len - window_size + 1, step_size)):
            u_id += 1
            window_start = i
            window_end = i + window_size
            window_seq = str(seq[window_start:window_end])
            windows.append(
                (u_id, seq_id, window_start, window_end, window_seq, '+'))
            revcomp = str(Seq(window_seq).reverse_complement())
            windows.append(
                (u_id, seq_id, window_start, window_end, revcomp, '-'))
        return windows

    windows = []
    for seq_id in ref_genome.keys():
        seq_windows = extract_windows_from_sequence(seq_id)
        windows.extend(seq_windows)

    # Convert windows to a Pandas DataFrame
    df = pd.DataFrame(
        windows, columns=['u_id', 'seq_id', 'start', 'end', 'seq', 'strand'])

    return df

def df_to_fasta(df: pd.DataFrame, filename: str, train_stat="testing") -> None: #TODO: This could be written to work faster
    """This function writes a Pandas DataFrame to a FASTA file.

    Args:
        df (pd.DataFrame): This is the Pandas DataFrame to be written to a FASTA file.
        filename (str): This is the name of the FASTA file to be written.
        train_stat (str, optional): This is train or test status. It is required by iLearnPlus, but doesn't affect our program functionality. Defaults to "testing".

    Returns:
        None: This function does not return anything.
    """    
    for i in range(len(df)):
        with open(f'{filename}', 'a') as f:
            head = str(df["u_id"][i]) + "_" + str(df["seq_id"][i]) + "_" + str(
                df["start"][i]) + "_" + str(df["end"][i]) + "_" + str(
                    df["strand"][i])
            f.write(f'>{head}|{"-1"}|{train_stat}\n')
            f.write(f'{df["seq"][i]}\n')


def csv_reader_low(path):
    df_test = pd.read_csv(f'{path}', nrows=100)
    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    # if bin or PS2 is in the name of the column float32_cols, it is a uint8
    for col in float32_cols.keys():
        if "bin" in col or "PS2" in col:
            float32_cols[col] = np.uint8

    float32_cols['SampleName'] = str
    float32_cols['label'] = bool
    df = pd.read_csv(f'{path}', engine='pyarrow', dtype=float32_cols)
    return df


def join_files(path, i):
    df = csv_reader_low(path + f"/ENAC-{i}-0.csv")
    lens = len(df.columns) - 2
    for file in os.listdir(path):
        if file.endswith(f"{i}.csv") and (not file.endswith(f"ENAC-{i}.csv")):
            # read the file
            df1 = csv_reader_low(path + "/" + file)
            df1 = df1.drop(['label'], axis=1)
            # append the file name to the all column names except the first two columns
            df1.columns = df1.columns[:2].tolist() + [
                file.replace(".csv", '') + "_" + col for col in df1.columns[2:]
            ]

            lens += len(df1.columns) - 2
            # join the files
            col_name = file.replace(".csv", '')
            print("Merging features from bathces", col_name)
            df = pd.merge(
                df,
                df1,
                how='outer',
                on=['SampleName'],
            )
    return df, lens


def col_dropper(path_to_shap, path_to_feature_imp, df):
    shap_importance = pd.read_csv(path_to_shap)
    feature_importances = pd.read_csv(path_to_feature_imp)

    # select 80 percent of feature with quantile to keep
    shap_importance_to_keep = shap_importance[
        shap_importance['feature_importance_vals'] >
        shap_importance['feature_importance_vals'].quantile(0.2)]
    feature_importance_to_keep = feature_importances[
        feature_importances['importance'] >
        feature_importances['importance'].quantile(0.2)]
    # intersection of the two
    features_to_keep = list(
        set(shap_importance_to_keep['col_name']).intersection(
            set(feature_importance_to_keep['feature'])))
    features_to_keep = [
        x for x in features_to_keep if any(c in x for c in [
            'Geary', 'NMBroto', 'PseKNC', 'Z_curve_144bit', 'Z_curve_9bit',
            'ENAC'
        ])
    ]
    # drop the columns that are not in the intersection
    df = df.drop([col for col in df.columns if col not in features_to_keep],
                 axis=1)
    return df


def read_csv_low(file, data_path, input_dim):
    df_test = pd.read_csv(f'{data_path}', nrows=100)
    dtype_cols = [c for c in df_test if df_test[c].dtype == "float64"]

    if file in ['PS2.csv', 'binary.csv']:
        dtype_cols = {c: np.int8 for c in dtype_cols}
    else:
        dtype_cols = {c: np.float32 for c in dtype_cols}

    x = pd.read_csv(
        f'{data_path}',
        dtype=dtype_cols,
        # engine = 'pyarrow',
        #  nrows=80000, #e################################################# comment for production
    )

    sample_names = x['SampleName']
    x.drop(columns=['SampleName', 'label'], inplace=True)

    reshaper_dim = list(input_dim[:])
    reshaper_dim.insert(0, len(x))
    # reshaper_dim.append(1)
    reshaper_dim = tuple(reshaper_dim)
    x = x.values.reshape(reshaper_dim)

    return x, sample_names

@terminator_filter_decorator
def process_and_merge_dataframe(genome_file: str,
                                df: pd.DataFrame,
                                filter_threshold: float,
                                file_format: str = file_format) -> pd.DataFrame:
    """
    Processes a DataFrame with BacTermFinder data.

    For FASTA, the expected format is:
      "X_<chrom>_<start>_<end>_<strand>"
    For GenBank (gb), two formats are supported:
      1) "1_NZ_CP017481.1_0_101_+"
      2) "CP017481.1:150-300(+)"

    Steps of the function:
      1. Extracts the chromosome, start and end coordinates, and strand from SampleName.
      2. Filters the dataset by 'probability_mean' (keeps rows where the value is > filter_threshold).

    Arguments:
      genome_file      : Used in decorator for GenBank-based filtering.
      df               : The input DataFrame containing the 'SampleName' column and prediction columns.
      filter_threshold : Threshold for filtering by 'probability_mean'.
      file_format      : The format of SampleName ('fasta' or 'genbank').

    Returns:
      A DataFrame of **raw** terminator candidates with columns:
      ['chrom','strand','start','end','probability_binary',
       'probability_ENAC','probability_PS2','probability_NCP','probability_mean']
      — ready for post-processing by the decorator.
    """
    df = df.copy()

    # Parsing SampleName
    if file_format.lower() == 'fasta':
        splits = df['SampleName'].str.split('_')
        df['chrom']  = splits.str[1]
        df['start']  = splits.str[2].astype(int)
        df['end']    = splits.str[3].astype(int)
        df['strand'] = splits.str[4]

    elif file_format.lower() == 'genbank':
        import re
        def parse_gb(name: str):
            parts = name.split('_')
            if len(parts) == 6:
                chrom = parts[1] + "_" + parts[2]
                return chrom, int(parts[3]), int(parts[4]), parts[5]
            if len(parts) == 5:
                return parts[1], int(parts[2]), int(parts[3]), parts[4]
            pat = re.compile(r'^(?P<chrom>[\w\.]+):(?P<start>\d+)[\-:](?P<end>\d+)\((?P<strand>[\+\-])\)')
            m = pat.search(name)
            if not m:
                raise ValueError(f"Can't parse GenBank SampleName: {name}")
            return (m.group('chrom'),
                    int(m.group('start')),
                    int(m.group('end')),
                    m.group('strand'))
        parsed = df['SampleName'].apply(parse_gb)
        df[['chrom','start','end','strand']] = pd.DataFrame(parsed.tolist(), index=df.index)

    else:
        raise ValueError(f"Unknown format: {file_format}")

    # We leave only the threshold for average probability
    df = df[df['probability_mean'] > filter_threshold].reset_index(drop=True)

    # sorting for stability
    df = df.sort_values(['chrom','strand','start']).reset_index(drop=True)

    # IMPORTANT: There is NO merging or aggregation here - the decorator will handle that.
    return df


def annotate_rho_dependent_terminator(df: pd.DataFrame, genome_file: str) -> None:
    """
    This function enhances the original GenBank file with annotations for a Rho-dependent terminator.
    In addition to the dataframe with coordinates and probabilities, the function receives the path to
    the original file (for example, passed as sys.argv[1]). As a result, a new file is created – a copy
    of the original with the added annotations.

    Expected dataframe format (each row corresponds to one terminator signal):
        chrom; strand; start; end; probability_binary; probability_ENAC; probability_PS2; probability_NCP; probability_mean

    For each row, a SeqFeature object is created with:
        - coordinates (start, end);
        - strand/direction (strand: '+' corresponds to 1, '-' corresponds to -1);
        - additional data (a dictionary of probabilities).

    Then, the original GenBank record from genome_file is updated:
        - The annotation date is updated to the current date,
        - New features from the dataframe are added to any existing features.

    The new file is saved in GenBank format with a name generated by appending the suffix `_annotated` to the original file name.
    """
    # Read the original GenBank file
    record = SeqIO.read(genome_file, file_format)

    # Update some annotation fields (e.g., the date)
    record.annotations["date"] = datetime.now().strftime("%d-%b-%Y").upper()

    # Create a list of new features based on the dataframe
    new_features = []
    for idx, row in df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        # Determine the strand: '+' -> 1, '-' -> -1
        strand = 1 if row["strand"] == "+" else -1
        location = FeatureLocation(start, end, strand=strand)

        qualifiers = {
            "probability_binary": float(row["probability_binary"]),
            "probability_ENAC": float(row["probability_ENAC"]),
            "probability_PS2": float(row["probability_PS2"]),
            "probability_NCP": float(row["probability_NCP"]),
            "probability_mean": float(row["probability_mean"]),
            "direction": "+" if strand == 1 else "-"
        }

        feature = SeqFeature(location=location, type="terminator", qualifiers=qualifiers)
        new_features.append(feature)

    # Add the new features to any existing ones (if present)
    record.features.extend(new_features)

    # Generate the new file name: take the original file name without its extension and add the "_annotated" suffix
    if "." in genome_file:
        base, ext = genome_file.rsplit(".", 1)
        new_filename = f"{base}_annotated.{ext}"
    else:
        new_filename = f"{genome_file}_annotated.gb"

    # Write the updated record to a new GenBank file
    with open(new_filename, "w") as output_handle:
        SeqIO.write(record, output_handle, file_format)

    print(f"Annotated file saved as: {new_filename}")


# Example usage:
# import sys
# genome_file = sys.argv[1]
# df = pd.read_csv("path_to_your_dataframe.tsv", sep=";")
# annotate_rho_dependent_terminator(df, genome_file)


if __name__ == '__main__':
    # time it
    start_time = time.time()
    ############################################ Sys inputs ########################################################
    if len(sys.argv) < 2:
        print(
            "ERROR: Enter file path (optional hyperparameters: [sliding window step] [prefix for output files] [Feature generation batch size] [threshold])",
            file=sys.stderr)
        input("Press Enter to exit...")
        sys.exit(1)
    genome_file = sys.argv[1]  # genome file name in genbank format
    step_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3  # step size for sliding window (aka stride size)
    output_file = sys.argv[3] if len(sys.argv) > 3 else "out"  # output file name
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 10000  # batch size for iLearnPlus feature generation
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.95  #the lowest value taken into account in the output sample of final values
    WINDOW_SIZE = 101  # window size for sliding windows, fixed to 101
    # get the sequences
    print("\n")
    print("Sliding windows generation started\n")
    if os.path.exists('Sample.csv'):
        os.remove('Sample.csv')
    df_slide = extract_sliding_windows(genome_file, WINDOW_SIZE, step_size)
    df_slide.to_csv(output_file.split(".")[0] + f'_{genome_file}_sliding_windows.csv',
                    index=False)
    print("\nSliding windows generated")
    ############################################  iLearnPlus ########################################################
    if os.path.exists('df_sample.fasta'):
        os.remove('df_sample.fasta')
    print("\n")
    print("iLearnPlus biological feature generation started")
    # convert the dataframe to fasta format
    df_to_fasta(df_slide.reset_index(drop=True), "df_sample.fasta", "training")

    # remove output folder if it exists
    if os.path.exists('output_sample'):
        shutil.rmtree('output_sample')

    # if os.path.exists('output_sample_merged'):
    #     os.remove('output_sample_merged')

    # generate features
    os.system('python iLearnPlus/util/FileProcessing.py ' +
              'df_sample.fasta' + ' ' + str(batch_size) + ' ' + '16' + ' ' +
              'output_sample')

    files = os.listdir('output_sample')
    
    number_of_batches = len(df_slide) // batch_size
    print("number_of_batches", number_of_batches )
    print("\niLearnPlus biological feature generation finished")
    endtime = time.time()
    print("Feature generation took", endtime - start_time, "s.")

    ############################################ Loading and prediction ########################################################

    # check gpu exists
    if os.system('nvidia-smi') == 0:
        gpu_exists = 1
        print('###### GPU exists, predicting with GPU ########')
    else:
        gpu_exists = 0
        print('###### GPU doesn\'t exists, predicting with CPU  ########')

    input_dim_dict = {
        'ENAC.csv': (97, 4),
        'PS2.csv': (100, 16),
        'NCP.csv': (101, 3),
        'binary.csv': (101, 4),
    }

    df = pd.DataFrame()
    data_path = 'output_sample/'
    print('\nLoading data')
    # files = os.listdir(data_path)
    for embedding in input_dim_dict.keys():
        # load deep learning model
        print(f"\nLoading Deep Learning Model {embedding} \n")
        model = load_model(f'deep-bactermfinder_3cnn_1d_1cat_reduced_10x_{embedding}_saved_model.h5')

        embedding_wo_csv = embedding.split('.csv')[0]
        out_embed = pd.DataFrame(columns=['SampleName', f'probability_{embedding_wo_csv}'])
        
        for batch in range(number_of_batches + 1):
            dp = f'{data_path}{embedding_wo_csv}-{batch}.csv'
            x, sample_info = read_csv_low(embedding, dp, input_dim_dict[embedding])
            print(f'x shape of {embedding} is: {x.shape}')

            print("Predicting sequences")
            with tf.device('/gpu:0'):
                y_pred = model.predict(x)

            # appending the results
            print("Appending the results")

            sample_info = sample_info.reset_index()
            y_pred = pd.DataFrame(y_pred, )
            y_pred[f'probability_{embedding_wo_csv}'] = y_pred[0]
            
            y_pred = pd.concat([sample_info, y_pred], axis=1)

            out_embed = pd.concat([out_embed, y_pred],ignore_index=True)
            del x
            del y_pred
            del sample_info

        # We save the results of the neural network in memory instead of writing to CSV.
        # We use a global dictionary to store dataframes by the embedding key.
        if 'results_in_memory' not in globals():
            results_in_memory = {}

        # Process the current out_embed the same way as before:
        out_embed.drop(columns=['index', 0], inplace=True)
        # Save a copy to a dictionary by key (for example, 'binary' or another one specified by embedding_wo_csv)
        results_in_memory[str(embedding_wo_csv)] = out_embed.copy()

        # Free up memory as in the original code
        del out_embed
        del model

        endtime_pred = time.time()
        print("Predicting sequences took", endtime_pred - endtime, "s.")

        # =======================================================================
        # MERGE RESULTS
        # Instead of reading files, select the base dataframe from the dictionary.
        # The merge logic does not change: add columns to the base dataframe
        # with a probability from each of the intermediate dataframes.
        # =======================================================================

        # Select the base key; it matches the embedding_wo_csv that was used (e.g. for 'binary')
        base_key = str(embedding_wo_csv)
        if base_key not in results_in_memory:
            raise ValueError(f"No data found for base embedding: {base_key}")

        df = results_in_memory[base_key].copy()

        # For all embeddings other than 'binary.csv', add the corresponding column.
        for embedding in input_dim_dict.keys():
            if embedding != 'binary.csv':
                # From the name we leave only the part without the extension (analogous to embedding.split('.csv')[0])
                emb_key = embedding.replace('.csv', '')
                if emb_key in results_in_memory:
                    # It is assumed that each dataframe already has a column with a name like 'probability_{emb_key}'
                    df[f'probability_{emb_key}'] = results_in_memory[emb_key][f'probability_{emb_key}']
                else:
                    print(f"Warning: data for embedding '{emb_key}' not found.")

        # Calculate the average value for columns containing 'probability'
        prob_cols = [col for col in df.columns if 'probability' in col]
        df['probability_mean'] = df[prob_cols].mean(axis=1)
        # The final dataframe df contains the results of all neural networks with the added column
        # average value. Further operations can be performed with df in memory.
    ############################################ timing and done ########################################################
    # Outputting a dataframe to a table
#   df.to_csv('TESTDATAFRAME.csv', sep=';', index=False)
    processedDF = process_and_merge_dataframe(genome_file=genome_file,df=df, filter_threshold=threshold)
    processedDF.to_csv('PROCESSEDDF.csv', sep=';', index=False)
    annotate_rho_dependent_terminator(processedDF, genome_file)
    del df
    del processedDF
    endtime = time.time()
    print("Totally, it took", endtime - start_time,
          f"s to find the terminators of a genome file that has {len(df_slide)} windows")
    print('Done')
