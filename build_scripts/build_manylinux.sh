#!/bin/bash

# clean up build env
[ -d build ] && rm -rf build
[ -d dist ] && rm -rf dist
[ -d wheelhouse ] && rm -rf wheelhouse
[ -d fixed_wheelhouse ] && rm -rf fixed_wheelhouse
rm *.so

echo 'Checking patchelf version ... '

export PATCHELF_BIN=/usr/local/bin/patchelf
patchelf_version=`$PATCHELF_BIN --version`
echo "patchelf version: " $patchelf_version
if [[ "$patchelf_version" == "patchelf 0.9" ]]; then
    echo "Your patchelf version is too old. Please use version >= 0.10."
    exit 1
fi

echo 'Building wheel ...'

python3 setup.py sdist bdist_wheel
org_wheel=$(find dist/ -type f -iname 'fastseq*linux*.whl')

echo 'Repairing the wheel as manylinux wheel ...'

PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
PLAT_FORM='manylinux2014_x86_64'
export LD_LIBRARY_PATH=$PYTHON_SITE_PACKAGES/torch/lib/:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
auditwheel repair --plat $PLAT_FORM -w wheelhouse $org_wheel

echo 'Fixing the wheel repaired by autitwheel ...'

manylinux_wheel=$(find wheelhouse/ -type f -iname 'fastseq*linux*.whl')
mkdir fixed_wheelhouse
unzip $manylinux_wheel -d fixed_wheelhouse

ngram_so_file=fixed_wheelhouse/ngram_repeat_block_cuda.so
fastseq_libs=fixed_wheelhouse/fastseq.libs
echo "Before fixing: "
patchelf --print-needed $ngram_so_file
patchelf --print-rpath $ngram_so_file

NON_DEP_LIBS=('libc10' 'libtorch' 'libtorch_cpu' 'libtorch_python' 'libc10_cuda' 'libtorch_cuda')
patchelf --print-needed $ngram_so_file| while read lib_sha256_so_file; do
    lib_so_name=$(echo $lib_sha256_so_file | cut -d- -f1)
    if [[ " ${NON_DEP_LIBS[@]} " =~ " $lib_so_name " ]]; then
        org_lib_so_file=$(echo $lib_sha256_so_file| sed -e 's/\-.*\./\./g');
        patchelf --replace-needed $lib_sha256_so_file $org_lib_so_file $ngram_so_file
        rm $fastseq_libs/$lib_sha256_so_file
    fi
done

echo "After fixing: "
patchelf --print-needed $ngram_so_file
patchelf --print-rpath $ngram_so_file

fname_with_sha256() {
    HASH=$(sha256sum $1 | cut -c1-8)
    DIRNAME=$(dirname $1)
    BASENAME=$(basename $1)
    if [[ $BASENAME == "libnvrtc-builtins.so" ]]; then
        echo $1
    else
        INITNAME=$(echo $BASENAME | cut -f1 -d".")
        ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
        echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    fi
}

make_wheel_record() {
    FPATH=$1
    if echo $FPATH | grep RECORD >/dev/null 2>&1; then
        # if the RECORD file, then
        echo "$FPATH,,"
    else
        HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
        FSIZE=$(ls -nl $FPATH | awk '{print $5}')
        echo "$FPATH,sha256=$HASH,$FSIZE"
    fi
}

echo 'Updating RECORD ...'

cd fixed_wheelhouse
record_file=fastseq*dist-info/RECORD
record_file=$(find . -type f -iname 'RECORD')
rm $record_file

find * -type f | while read fname; do
    echo $(make_wheel_record $fname) >> $record_file
done

fixed_manylinux_wheel_path=fixed_wheelhouse/$(basename $manylinux_wheel)
zip -rq $(basename $fixed_manylinux_wheel_path) .
shopt -s extglob
rm -rf !($(basename $fixed_manylinux_wheel_path))

echo "Fastseq manylinux wheel is generated at: $fixed_manylinux_wheel_path"
