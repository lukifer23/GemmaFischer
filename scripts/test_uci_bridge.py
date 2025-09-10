#!/usr/bin/env python3
"""
Test script for UCI Bridge

This script tests the UCI bridge functionality with sample commands.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.uci_bridge import UCIBridge

def test_uci_bridge():
    """Test the UCI bridge with sample commands"""
    
    print("Testing UCI Bridge...")
    
    # Initialize the bridge
    bridge = UCIBridge()
    
    # Test commands
    test_commands = [
        "uci",
        "isready",
        "setoption name Mode value tutor",
        "setoption name Style value fischer",
        "position startpos",
        "position startpos moves e2e4 e7e5",
        "go depth 5",
        "quit"
    ]
    
    print("\nRunning test commands:")
    print("=" * 50)
    
    for command in test_commands:
        print(f"Command: {command}")
        response = bridge.handle_uci_command(command)
        if response:
            print(f"Response: {response}")
        print("-" * 30)
    
    print("UCI Bridge test completed!")

if __name__ == "__main__":
    test_uci_bridge()
