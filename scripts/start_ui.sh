#!/bin/bash
#
# LitForge Web UI Launch Script
# =============================
# Starts the Streamlit web UI in the background
#
# Usage:
#   ./scripts/start_ui.sh          # Start form UI on default port 8502
#   ./scripts/start_ui.sh chat     # Start CHAT interface on port 8503
#   ./scripts/start_ui.sh 8504     # Start on custom port
#   ./scripts/start_ui.sh stop     # Stop the server
#   ./scripts/start_ui.sh status   # Check if running
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# Determine mode (chat or form)
MODE="form"
PORT=8502
UI_FILE="app.py"

for arg in "$@"; do
    case "$arg" in
        chat)
            MODE="chat"
            PORT=8503
            UI_FILE="chat.py"
            ;;
        stop|status|restart)
            ;;
        [0-9]*)
            PORT="$arg"
            ;;
    esac
done

PID_FILE="$LOG_DIR/litforge_ui_${MODE}.pid"
LOG_FILE="$LOG_DIR/litforge_ui_${MODE}.log"

# Create logs directory
mkdir -p "$LOG_DIR"

# Load Python module
module load python3 2>/dev/null || true

start_server() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "🔥 LitForge UI ($MODE mode) is already running (PID: $(cat $PID_FILE))"
        echo "   URL: http://$(hostname):$PORT"
        return 1
    fi
    
    echo "🔥 Starting LitForge Web UI ($MODE mode)..."
    echo "   Port: $PORT"
    echo "   Log: $LOG_FILE"
    
    cd "$PROJECT_DIR"
    
    nohup crun -p ~/envs/litforge python -m streamlit run \
        "src/litforge/ui/$UI_FILE" \
        --server.port "$PORT" \
        --server.headless true \
        --server.address 0.0.0.0 \
        > "$LOG_FILE" 2>&1 &
    
    echo $! > "$PID_FILE"
    
    # Wait a moment for startup
    sleep 3
    
    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "   ✅ Server started (PID: $(cat $PID_FILE))"
        echo ""
        echo "   Access at: http://$(hostname):$PORT"
        echo "   Or locally: http://localhost:$PORT"
        echo ""
        echo "   To stop: $0 stop"
    else
        echo "   ❌ Server failed to start. Check $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_server() {
    # Stop both modes
    for mode in form chat; do
        pid_file="$LOG_DIR/litforge_ui_${mode}.pid"
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            if kill -0 "$PID" 2>/dev/null; then
                echo "🛑 Stopping LitForge UI ($mode mode, PID: $PID)..."
                kill "$PID"
                rm -f "$pid_file"
                echo "   ✅ Server stopped"
            else
                echo "   Server not running (stale PID file)"
                rm -f "$pid_file"
            fi
        fi
    done
}

status_server() {
    for mode in form chat; do
        pid_file="$LOG_DIR/litforge_ui_${mode}.pid"
        if [ -f "$pid_file" ] && kill -0 $(cat "$pid_file") 2>/dev/null; then
            port=$([ "$mode" = "chat" ] && echo "8503" || echo "8502")
            echo "🔥 LitForge UI ($mode mode) is running"
            echo "   PID: $(cat $pid_file)"
            echo "   URL: http://$(hostname):$port"
        else
            echo "⚪ LitForge UI ($mode mode) is not running"
        fi
    done
}

case "$1" in
    stop)
        stop_server
        ;;
    status)
        status_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    *)
        start_server
        ;;
esac
