source /etc/profile

#样本文件
input_file=$1
#训练文件
output_train_file=$2
#测试文件
output_test_file=$3

echo $3
cp ${input_file} ${input_file}.bak

columns=$(head -n 1 ${input_file})
echo "${columns}" > "${output_train_file}"
echo "${columns}" > "${output_test_file}"

sed -i '1d' ${input_file}
shuf ${input_file} -o ${input_file}.shuf

length=$(wc -l ${input_file} | awk '{printf("%d", $1)}')

head_cure=$(echo ${length} | awk '{printf("%d", $1 * 0.9)}')
tail_cure=$(( ${length} - ${head_cure} - 1 ))

head -n ${head_cure} "${input_file}.shuf" >> "${output_train_file}"
tail -n ${tail_cure} "${input_file}.shuf" >> "${output_test_file}"

cp ${input_file}.bak ${input_file}