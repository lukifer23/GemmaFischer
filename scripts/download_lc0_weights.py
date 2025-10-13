#!/usr/bin/env python3
"""
Download LC0 network weights for GemmaFischer integration.

This script downloads and validates LC0 neural network weights
for use in the hybrid LLM + LC0 chess system.
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

class LC0WeightsDownloader:
    """Download and manage LC0 neural network weights."""
    
    # Known working LC0 network sources
    NETWORK_SOURCES = {
        'T60-3770': {
            'url': 'https://training.lczero.org/networks/512x15x8h-t60-3770/network-3770.pb.gz',
            'hash': None,  # We'll validate by testing with LC0
            'description': 'T60 network - good balance of strength and speed'
        },
        'T75-2380': {
            'url': 'https://training.lczero.org/networks/512x15x8h-t75-2380/network-2380.pb.gz', 
            'hash': None,
            'description': 'T75 network - stronger but slower'
        },
        # Add more networks as needed
    }
    
    def __init__(self, weights_dir: str = "models/lc0_weights"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
    def download_network(self, network_name: str, force: bool = False) -> bool:
        """Download a specific LC0 network."""
        if network_name not in self.NETWORK_SOURCES:
            print(f"❌ Unknown network: {network_name}")
            return False
            
        network_info = self.NETWORK_SOURCES[network_name]
        filename = f"{network_name}.pb.gz"
        filepath = self.weights_dir / filename
        
        # Check if already exists
        if filepath.exists() and not force:
            print(f"✅ {network_name} already exists at {filepath}")
            return self.validate_network(filepath)
            
        print(f"📥 Downloading {network_name} from {network_info['url']}")
        
        try:
            response = requests.get(network_info['url'], stream=True, timeout=300)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Simple progress indicator
                        if total_size > 0:
                            progress = int(50 * downloaded / total_size)
                            print(f"\r[{'█' * progress}{'.' * (50-progress)}] {downloaded}/{total_size} bytes", end='')
            
            print(f"\n✅ Downloaded {network_name} ({downloaded} bytes)")
            return self.validate_network(filepath)
            
        except Exception as e:
            print(f"❌ Failed to download {network_name}: {e}")
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
            return False
    
    def validate_network(self, filepath: Path) -> bool:
        """Validate that a downloaded network file is valid."""
        if not filepath.exists():
            return False
            
        # Check file size (networks should be > 100MB)
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb < 100:
            print(f"❌ Network file too small: {size_mb:.1f}MB")
            return False
            
        # Try to test with LC0 (basic validation)
        import subprocess
        try:
            result = subprocess.run([
                '/opt/homebrew/bin/lc0',
                f'--weights={filepath}',
                '--backend=metal',
                'backendbench',
                '--batches=1'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ Network validated: {filepath.name} ({size_mb:.1f}MB)")
                return True
            else:
                print(f"❌ Network validation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Network validation timed out")
            return False
        except Exception as e:
            print(f"❌ Network validation error: {e}")
            return False
    
    def list_available_networks(self):
        """List available networks for download."""
        print("Available LC0 Networks:")
        print("=" * 50)
        for name, info in self.NETWORK_SOURCES.items():
            print(f"• {name}: {info['description']}")
            print(f"  URL: {info['url']}")
        print()
    
    def download_default_network(self) -> Optional[Path]:
        """Download the recommended default network (T60)."""
        return self.download_network('T60-3770')

def main():
    """Main download script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download LC0 network weights')
    parser.add_argument('--network', default='T60-3770', 
                       choices=['T60-3770', 'T75-2380'],
                       help='Network to download')
    parser.add_argument('--list', action='store_true', 
                       help='List available networks')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if exists')
    parser.add_argument('--dir', default='models/lc0_weights',
                       help='Directory to store weights')
    
    args = parser.parse_args()
    
    downloader = LC0WeightsDownloader(args.dir)
    
    if args.list:
        downloader.list_available_networks()
        return
    
    success = downloader.download_network(args.network, args.force)
    
    if success:
        filepath = downloader.weights_dir / f"{args.network}.pb.gz"
        print(f"\\n🎉 Success! LC0 network ready at: {filepath}")
        print(f"   Size: {filepath.stat().st_size / (1024*1024):.1f}MB")
    else:
        print("\\n❌ Failed to download/validate network")
        sys.exit(1)

if __name__ == '__main__':
    main()
