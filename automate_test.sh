dir_name="shareable"
another_dir="../test"

if [[ -n "$1" ]]; then
    dir_name="$1"
fi

if [[ -n "$2" ]]; then
    another_dir="$2"
fi

scripts/obfuscate_workspace.sh . --output "$dir_name"

mkdir -p "$another_dir"
cp -r "$dir_name" "$another_dir/"

cd "$another_dir/$dir_name" || exit 1

installation/install.sh
echo "$PWD"

cd app || exit 1

chmod +x run_config.sh
chmod +x start_hands.sh

timeout 30 ./run_config.sh
RET_CONFIG=$?

timeout 40 ./start_hands.sh
RET_START=$?

echo "RET_CONFIG: $RET_CONFIG"
echo "RET_START: $RET_START"

if [[ $RET_START -eq 124 ]]; then
    RET_START=0
fi

if [[ $RET_CONFIG -eq 124 ]]; then
    RET_CONFIG=0
fi

if [[ $RET_CONFIG -ne 0 || $RET_START -ne 0 ]]; then
    echo "One or more tasks failed."
    exit 1
fi

echo "Do you want to delete the dir? Y/N"
read ans

if [[ "$ans" == "Y" || "$ans" == "y" ]]; then
    cd ../.. || exit 1
    rm -rf "$another_dir"
fi
