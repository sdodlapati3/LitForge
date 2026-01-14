#!/bin/bash
#
# LitForge MCP Server Launch Script
# ==================================
# Starts the MCP server for AI assistant integration
#
# Usage:
#   ./scripts/start_mcp.sh          # Start MCP server
#   ./scripts/start_mcp.sh stop     # Stop the server
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/litforge_mcp.pid"
LOG_FILE="$LOG_DIR/litforge_mcp.log"

mkdir -p "$LOG_DIR"
module load python3 2>/dev/null || true

start_server() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "🔥 LitForge MCP Server is already running (PID: $(cat $PID_FILE))"
        return 1
    fi
    
    echo "🔥 Starting LitForge MCP Server..."
    
    cd "$PROJECT_DIR"
    
    nohup crun -p ~/envs/litforge python -m litforge.mcp.server \
        > "$LOG_FILE" 2>&1 &
    
    echo $! > "$PID_FILE"
    sleep 2
    
    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "   ✅ MCP Server started (PID: $(cat $PID_FILE))"
        echo "   Log: $LOG_FILE"
    else
        echo "   ❌ Server failed to start. Check $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "🛑 Stopping LitForge MCP Server..."
            kill "$PID"
            rm -f "$PID_FILE"
            echo "   ✅ Server stopped"
        else
            rm -f "$PID_FILE"
        fi
    fi
}

case "$1" in
    stop) stop_server ;;
    restart) stop_server; sleep 2; start_server ;;
    *) start_server ;;
esac
