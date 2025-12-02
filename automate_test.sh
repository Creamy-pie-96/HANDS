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

cd "$another_dir/$dir_name/installation" || exit 1


./install.sh

echo "Do you want to delete the dir? Y/N"
read ans

if [[ "$ans" == "Y" || "$ans" == "y" ]]; then
    rm -rf "$another_dir"
fi
