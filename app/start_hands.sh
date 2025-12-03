#!/bin/bash
#
# HANDS Quick Start Script
# Launches the HANDS application with proper environment
#

# Minimal launcher output (suppressed verbose startup banners)


# Resolve the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is the parent of the app directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# project root is available in $PROJECT_ROOT (no verbose printing)

# Help function: display usage and examples
show_help() {
        cat <<'EOF'
HANDS Quick Start Script

Usage:
    ./app/start_hands.sh [--clean] [control-flags] [-- <forwarded to hands_app>]

Control flags (trigger app_control.py):
    --pause [true|false]     Set app_control.pause in config (bare --pause => true)
    --exit  [true|false]     Set app_control.exit in config (bare --exit => true)
    --config <path>          Path to config.json forwarded to app_control.py
    --status                 Show current app_control values (runs app_control.py --status)

Other flags are forwarded to HANDS app (`hands_app.py`) as-is, e.g.:
    --camera <index>
    --dry-run
    --config <path>

Examples:
    ./app/start_hands.sh --exit true
    ./app/start_hands.sh --pause true
    ./app/start_hands.sh --pause=false
    ./app/start_hands.sh --exit true --pause false
    ./app/start_hands.sh --config /path/to/config.json --exit true
    ./app/start_hands.sh --status

Behavior:
    - If any of --pause/--exit/--config/--status are present the script runs
        `app_control.py` with those flags before starting the HANDS app.
    - If `--exit true` is set, the script will run the control app, wait 5s, then
        exit WITHOUT starting the HANDS app.
    - If control-only flags are provided (no other non-control args), the script
        runs `app_control.py` and exits.
EOF
}

# If user asked for help, show it and exit
for arg in "$@"; do
        case "$arg" in
                -h|--help)
                        show_help
                        exit 0
                        ;;
        esac
done

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found in $(pwd)!"
    echo "   Please create one with: python3 -m venv .venv"
    exit 1
fi

# Parse arguments and support cleaning flags anywhere: c, -c, --clean
# We preserve the order of other args and forward them to hands_app.py
CLEAN=false
NEWARGS=()
for a in "$@"; do
    case "$a" in
        c|-c|--clean)
            CLEAN=true
            ;;
        *)
            NEWARGS+=("$a")
            ;;
    esac
done

# Keep original forwarded args copy for control detection/forwarding
ORIGARGS=("${NEWARGS[@]}")

# Extract control flags (--pause, --exit) from the forwarded args.
# We accept these forms:
#   --pause <value>
#   --pause=<value>
#   --exit <value>
#   --exit=<value>
# If the value is omitted (just `--pause`), we treat it as true.
PAUSE_FLAG=false
PAUSE_VAL=""
EXIT_FLAG=false
EXIT_VAL=""
FINALARGS=()
STATUS_FLAG=false

# Iterate over previously-collected NEWARGS and peel off control flags
i=0
while [ $i -lt ${#NEWARGS[@]} ]; do
    a="${NEWARGS[$i]}"
    case "$a" in
        --start)
            START_FLAG=true
            ;;
        --start=*)
            START_FLAG=true
            ;;
        --pause)
            # next token is value if present and not another flag
            next=$((i+1))
            if [ $next -lt ${#NEWARGS[@]} ]; then
                v="${NEWARGS[$next]}"
                if [[ "$v" != --* ]]; then
                    PAUSE_FLAG=true
                    PAUSE_VAL="$v"
                    i=$((i+1))
                else
                    PAUSE_FLAG=true
                    PAUSE_VAL=true
                fi
            else
                PAUSE_FLAG=true
                PAUSE_VAL=true
            fi
            ;;
        --pause=*)
            PAUSE_FLAG=true
            PAUSE_VAL="${a#--pause=}"
            ;;
        --exit)
            next=$((i+1))
            if [ $next -lt ${#NEWARGS[@]} ]; then
                v="${NEWARGS[$next]}"
                if [[ "$v" != --* ]]; then
                    EXIT_FLAG=true
                    EXIT_VAL="$v"
                    i=$((i+1))
                else
                    EXIT_FLAG=true
                    EXIT_VAL=true
                fi
            else
                EXIT_FLAG=true
                EXIT_VAL=true
            fi
            ;;
        --exit=*)
            EXIT_FLAG=true
            EXIT_VAL="${a#--exit=}"
            ;;
        --status)
            STATUS_FLAG=true
            ;;
        --status=*)
            STATUS_FLAG=true
            ;;
        --start)
            # handled above; noop
            ;;
        --start=*)
            # handled above; noop
            ;;
        *)
            FINALARGS+=("$a")
            ;;
    esac
    i=$((i+1))
done

# Replace NEWARGS with FINALARGS (these will be forwarded to hands_app)
NEWARGS=("")
unset NEWARGS
NEWARGS=()
for v in "${FINALARGS[@]}"; do
    NEWARGS+=("$v")
done

# We'll call the Python control script after we find the python executable.

if [ "$CLEAN" = true ]; then
    echo "Cleaning Python caches..."
    find "$PROJECT_ROOT" -type d -name "__pycache__" -print -exec rm -rf {} +
    find "$PROJECT_ROOT" -type f -name "*.pyc" -print -delete
    # Clear the terminal to remove noisy output from the cleanup
    clear
fi

# Locate venv python executable in a cross-platform way
if [ -x "$PROJECT_ROOT/.venv/bin/python3" ]; then
    PYEXEC="$PROJECT_ROOT/.venv/bin/python3"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYEXEC="$PROJECT_ROOT/.venv/bin/python"
else
    echo "❌ Python executable not found in .venv/bin/"
    echo "   Ensure the virtualenv is created and contains a python binary"
    exit 1
fi

# Starting HANDS application (minimal output)
# If user requested pause/exit control, call the app_control script first
if [ "$PAUSE_FLAG" = true ] || [ "$EXIT_FLAG" = true ] || [ "$STATUS_FLAG" = true ]; then
    # Build control args by forwarding any control-related tokens from ORIGARGS
    CTRL_ARGS=()
    i=0
    while [ $i -lt ${#ORIGARGS[@]} ]; do
        tok="${ORIGARGS[$i]}"
        case "$tok" in
            --pause|--pause=*|--exit|--exit=*|--status|--config|--config=*)
                # If it's of form --flag=value, pass through as-is
                if [[ "$tok" == *=* ]]; then
                    CTRL_ARGS+=("$tok")
                else
                    # If the token expects a value (like --config/--pause/--exit) and next token isn't another flag, include it
                    if [[ "$tok" == --status ]]; then
                        CTRL_ARGS+=("$tok")
                    else
                        next=$((i+1))
                        if [ $next -lt ${#ORIGARGS[@]} ]; then
                            nv="${ORIGARGS[$next]}"
                            if [[ "$nv" != --* ]]; then
                                CTRL_ARGS+=("$tok" "$nv")
                                i=$((i+1))
                            else
                                # No explicit value, treat as flag-only
                                CTRL_ARGS+=("$tok")
                            fi
                        else
                            CTRL_ARGS+=("$tok")
                        fi
                    fi
                fi
                ;;
        esac
        i=$((i+1))
    done

    # If no explicit --config was provided, set default config path for control
    has_config=false
    for t in "${CTRL_ARGS[@]}"; do
        if [[ "$t" == --config* ]]; then
            has_config=true
            break
        fi
    done
    if [ "$has_config" = false ]; then
        # Use absolute path to config relative to project root
        CTRL_ARGS+=(--config "$PROJECT_ROOT/source_code/config/config.json")
    fi

    # Determine the config path used by control so we can reuse it for resetting
    CTRL_CONFIG=""
    j=0
    while [ $j -lt ${#CTRL_ARGS[@]} ]; do
        t="${CTRL_ARGS[$j]}"
        if [[ "$t" == --config=* ]]; then
            CTRL_CONFIG="${t#--config=}"
            break
        elif [[ "$t" == --config ]]; then
            nextj=$((j+1))
            if [ $nextj -lt ${#CTRL_ARGS[@]} ]; then
                CTRL_CONFIG="${CTRL_ARGS[$nextj]}"
                break
            fi
        fi
        j=$((j+1))
    done
    if [ -z "$CTRL_CONFIG" ]; then
        CTRL_CONFIG="$PROJECT_ROOT/source_code/config/config.json"
    fi

    # Invoke control script silently (it prints its own confirmations/errors)
    "$PYEXEC" "$PROJECT_ROOT/source_code/scripts/app_control.py" "${CTRL_ARGS[@]}"

    # If exit flag set to a truthy value, wait 5s then exit without starting HANDS
    if [ "$EXIT_FLAG" = true ]; then
        # Normalize EXIT_VAL to lowercase
        ev=$(echo "${EXIT_VAL}" | tr '[:upper:]' '[:lower:]')
        if [ -z "$ev" ] || [ "$ev" = "true" ] || [ "$ev" = "1" ] || [ "$ev" = "yes" ]; then
            echo "Exit requested via --exit; waiting 5s then exiting..."
            sleep 5
            # Reset the exit flag using the same config the control invocation used
            "$PYEXEC" "$PROJECT_ROOT/source_code/scripts/app_control.py" --exit false --config "$CTRL_CONFIG"
            exit 0
        fi
    fi

    # If the invocation was control-only (no non-control args), exit after running control
    if [ ${#NEWARGS[@]} -eq 0 ]; then
        echo "Control-only invocation detected; exiting after app_control." 
        exit 0
    fi
fi

# Run as a module from the root directory
"$PYEXEC" -m source_code.app.hands_app "${NEWARGS[@]}"
