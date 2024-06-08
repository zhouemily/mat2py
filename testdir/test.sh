echo "testcases for  stabcheck36_trio_CM2.py script"
echo "show usage help menu:"
./stabcheck36_trio_CM2.py -h

echo "result and stability_history_2024.txt files are generated in ./logdir"
echo "test with 64 random Dicom files with prefix: T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0004 "
testcase="./stabcheck36_trio_CM2.py -d test_dir_d004/test_64_files"
echo $testcase
echo ""
$testcase
echo "test with 128 random Dicom files with prefix: T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0004 "
testcase="./stabcheck36_trio_CM2.py -d test_dir_d004/test_128_files"
echo $testcase
echo ""
$testcase
echo "test with 256 random Dicom files with prefix: T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0004 "
testcase="./stabcheck36_trio_CM2.py -d test_dir_d004/test_256_files"
echo $testcase
echo ""
$testcase
echo "test with 64 random Dicom files with prefix: T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0005 "
testcase="./stabcheck36_trio_CM2.py -d test_dir_d005/test_64_files"
echo $testcase
echo ""
$testcase
echo "test with 128 random Dicom files with prefix: T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0005 "
testcase="./stabcheck36_trio_CM2.py -d test_dir_d005/test_128_files"
echo $testcase
echo ""
$testcase
echo "test with 256 random Dicom files with prefix: T_STAB_12_8_2023.MR.SUTTON_SEQTESTING.0005 "
testcase="./stabcheck36_trio_CM2.py -d test_dir_d005/test_256_files"
echo $testcase
echo ""
$testcase
echo "results and stability_history_2024.txt files are generated in ./logdir"
