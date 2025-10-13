#!/usr/bin/env python3
"""
LC0 Network Verification and Repair Tool

Verifies the integrity of LC0 network files and provides repair options.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

class LC0NetworkVerifier:
    """Verify and repair LC0 network files."""
    
    def __init__(self, weights_dir: str = "models/lc0_weights"):
        self.weights_dir = Path(weights_dir)
        self.lc0_path = "/opt/homebrew/bin/lc0"
        
    def verify_network_file(self, network_path: Path) -> Dict[str, Any]:
        """Verify a network file's integrity."""
        result = {
            'path': str(network_path),
            'exists': network_path.exists(),
            'valid': False,
            'size_mb': 0.0,
            'file_type': 'unknown',
            'lc0_parseable': False,
            'error': None
        }
        
        if not network_path.exists():
            result['error'] = "File does not exist"
            return result
        
        # Check file size
        size_bytes = network_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        result['size_mb'] = round(size_mb, 2)
        
        # Check if file is too small (networks should be > 100MB)
        if size_mb < 100:
            result['error'] = f"File too small ({size_mb:.1f}MB). Valid networks are 100MB+"
        
        # Check file type
        try:
            file_check = subprocess.run(
                ['file', str(network_path)],
                capture_output=True,
                text=True,
                timeout=5
            )
            file_type = file_check.stdout.split(':', 1)[1].strip()
            result['file_type'] = file_type
            
            # Check if it's actually gzipped protobuf
            if 'HTML' in file_type or 'ASCII' in file_type:
                result['error'] = f"Not a valid network file. File type: {file_type}"
            elif 'gzip' not in file_type.lower():
                result['error'] = f"Not a gzip file. File type: {file_type}"
        except Exception as e:
            result['error'] = f"Could not determine file type: {e}"
        
        # Test with LC0 if available
        if Path(self.lc0_path).exists():
            try:
                lc0_test = subprocess.run(
                    [self.lc0_path, 'bench', f'--weights={network_path}', '--backend=metal'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Check if LC0 could parse it
                if 'unparseable' in lc0_test.stdout.lower() or 'error' in lc0_test.stdout.lower():
                    result['lc0_parseable'] = False
                    result['error'] = "LC0 cannot parse this file"
                elif lc0_test.returncode == 0:
                    result['lc0_parseable'] = True
                    result['valid'] = True
                    result['error'] = None
            except subprocess.TimeoutExpired:
                result['error'] = "LC0 test timed out"
            except Exception as e:
                result['error'] = f"LC0 test failed: {e}"
        
        return result
    
    def scan_all_networks(self) -> Dict[str, Any]:
        """Scan all network files in the weights directory."""
        if not self.weights_dir.exists():
            return {
                'weights_dir': str(self.weights_dir),
                'exists': False,
                'networks': []
            }
        
        network_files = list(self.weights_dir.glob("*.pb.gz"))
        results = {
            'weights_dir': str(self.weights_dir),
            'exists': True,
            'network_count': len(network_files),
            'networks': []
        }
        
        for network_file in network_files:
            verification = self.verify_network_file(network_file)
            results['networks'].append(verification)
        
        return results
    
    def print_report(self, results: Dict[str, Any]) -> None:
        """Print a human-readable verification report."""
        print("=" * 80)
        print("LC0 NETWORK VERIFICATION REPORT")
        print("=" * 80)
        print()
        
        print(f"Weights Directory: {results['weights_dir']}")
        print(f"Directory Exists: {results['exists']}")
        print()
        
        if not results['exists']:
            print("❌ Weights directory does not exist!")
            print(f"   Create it with: mkdir -p {results['weights_dir']}")
            return
        
        print(f"Network Files Found: {results['network_count']}")
        print()
        
        if results['network_count'] == 0:
            print("❌ No network files found!")
            print("   Download a network with: python scripts/download_lc0_weights.py")
            return
        
        valid_count = 0
        for network in results['networks']:
            filename = Path(network['path']).name
            print(f"Network: {filename}")
            print(f"  Size: {network['size_mb']:.1f} MB")
            print(f"  Type: {network['file_type']}")
            
            if network['valid']:
                print(f"  Status: ✅ VALID")
                valid_count += 1
            else:
                print(f"  Status: ❌ INVALID")
                print(f"  Error: {network['error']}")
            print()
        
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Networks: {results['network_count']}")
        print(f"Valid Networks: {valid_count}")
        print(f"Invalid Networks: {results['network_count'] - valid_count}")
        print()
        
        if valid_count == 0:
            print("❌ NO VALID NETWORKS FOUND")
            print()
            print("RECOMMENDED ACTIONS:")
            print("1. Remove corrupted files:")
            print(f"   rm {results['weights_dir']}/*.pb.gz")
            print()
            print("2. Download a valid network:")
            print("   python scripts/download_lc0_weights.py --network T60-3770")
            print()
        else:
            print("✅ Valid LC0 network(s) found!")
    
    def cleanup_corrupted_files(self) -> int:
        """Remove corrupted network files."""
        if not self.weights_dir.exists():
            return 0
        
        removed = 0
        for network_file in self.weights_dir.glob("*.pb.gz"):
            verification = self.verify_network_file(network_file)
            if not verification['valid']:
                print(f"Removing corrupted file: {network_file.name}")
                network_file.unlink()
                removed += 1
        
        return removed

def main():
    """Main verification routine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify LC0 network integrity',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove corrupted network files'
    )
    parser.add_argument(
        '--dir',
        default='models/lc0_weights',
        help='Weights directory to check (default: models/lc0_weights)'
    )
    
    args = parser.parse_args()
    
    verifier = LC0NetworkVerifier(args.dir)
    
    if args.cleanup:
        print("Cleaning up corrupted network files...")
        removed = verifier.cleanup_corrupted_files()
        print(f"Removed {removed} corrupted file(s)")
        print()
    
    # Run verification
    results = verifier.scan_all_networks()
    verifier.print_report(results)
    
    # Exit with error code if no valid networks found
    valid_networks = sum(1 for n in results.get('networks', []) if n['valid'])
    sys.exit(0 if valid_networks > 0 else 1)

if __name__ == '__main__':
    main()

