# Modification for the BacTermFinder program.<br>
**Link to the original:** 
[https://github.com/BioinformaticsLabAtMUN/BacTermFinder]<br>
**The command to start the program is replaced by is replaced by:** 
```bash
python genome_scan.py [genome fasta file] [sliding window step] [output file prefix] [feature generation batch size] [threshold] > log.out<br>
```
**recommended command**
```bash
python genome_scan.py annotatedHI.gb 3 out 10000 0.95 > log.out
```
---
List of main modifications:
1. The program is configured to work with files of the .gb format;
2. Filtering by threshold is performed inside the main code;
3. An algorithm for filtering unlikely terminators based on an annotated genome is added;
4. Automatic annotation is performed in the .gb file.
---
*To activate the mod, you just need to replace the original genome_scan.py file with the file from the mod.*
