SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root is parent of #dev
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

dir_name="shareable"
another_dir="../test"

if [[ -n "$1" ]]; then
    dir_name="$1"
fi

if [[ -n "$2" ]]; then
    another_dir="$2"
fi

"$REPO_ROOT/scripts/obfuscate_workspace.sh" "$REPO_ROOT" --output "$dir_name"

mkdir -p "$another_dir"
cp -r "$dir_name" "$another_dir/"

cd "$another_dir/$dir_name" || exit 1

# Run the Python installer from the copied test shareable directory
if [[ -f "installation/install.py" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        echo "Running installation/install.py..."
        python3 "installation/install.py" || { echo "installation/install.py failed"; exit 1; }
    else
        echo "Error: python3 not found in PATH"
        exit 1
    fi
else
    echo "Error: installation/install.py not found in copied shareable directory"
    exit 1
fi
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
