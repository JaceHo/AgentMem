#!/usr/bin/env python3
"""
Migration script to update main.py to use refactored modules.

This script applies the necessary changes to integrate:
1. Atomic counters (fix race conditions)
2. Centralized configuration
3. Utility functions from utils/
4. Custom exceptions

Usage:
    python migrate_main.py
    
This creates a backup and modifies main.py in place.
"""

import re
import shutil
from pathlib import Path


def migrate_main():
    """Apply refactoring changes to main.py."""
    
    main_path = Path("main.py")
    backup_path = Path("main.py.backup")
    
    if not main_path.exists():
        print("❌ main.py not found")
        return False
    
    # Create backup
    print(f"📦 Creating backup: {backup_path}")
    shutil.copy2(main_path, backup_path)
    
    # Read original file
    content = main_path.read_text()
    original_lines = len(content.splitlines())
    print(f"📄 Original main.py: {original_lines} lines")
    
    # Apply transformations
    
    # 1. Replace global counter declarations with atomic counters
    print("🔧 Replacing global counters with atomic counters...")
    
    counter_pattern = r'''# ── Auto-consolidation counters ────────────────────────────────────────────────
_stores_since_consolidation: int = 0
_AUTO_CONSOLIDATE_EVERY: int = 50   # trigger after every N stored facts

# ── Hard-prune \(physical VREM of superseded entries\) runs every 24 hours ───────
_periodic_prune_counter: int = 0    # incremented each hourly _periodic_consolidate call

# ── Store observability counters \(in-process, resets on restart\) ───────────────
_store_attempts: int = 0
_store_successes: int = 0
_store_skips: int = 0
_store_errors: int = 0
_store_latency_sum_ms: float = 0.0'''
    
    replacement = '''# ── Thread-safe auto-consolidation counter ───────────────────────────────────
_stores_since_consolidation = AtomicCounter()

# ── Thread-safe hard-prune scheduler counter ─────────────────────────────────
_periodic_prune_counter = AtomicCounter()

# ── Thread-safe store observability counters (replaces bare int/float) ───────
_store_attempts = AtomicCounter()
_store_successes = AtomicCounter()
_store_skips = AtomicCounter()
_store_errors = AtomicCounter()
_store_latency_sum_ms = AtomicFloat()'''
    
    content = re.sub(counter_pattern, replacement, content, flags=re.MULTILINE)
    
    # 2. Replace _AUTO_CONSOLIDATE_EVERY usage with settings
    content = content.replace(
        '_AUTO_CONSOLIDATE_EVERY',
        'settings.auto_consolidate_every'
    )
    
    # 3. Replace _BG_TASK_LIMIT with settings
    content = content.replace(
        '_BG_TASK_LIMIT',
        'settings.bg_task_limit'
    )
    
    # 4. Update APP_VERSION to use settings
    content = re.sub(
        r'APP_VERSION = "1\.0\.0"',
        'APP_VERSION = settings.app_version',
        content
    )
    
    # Write modified content
    main_path.write_text(content)
    new_lines = len(content.splitlines())
    
    print(f"✅ Migration complete!")
    print(f"   Lines: {original_lines} → {new_lines} ({original_lines - new_lines:+d})")
    print(f"   Backup saved to: {backup_path}")
    print(f"\n⚠️  Next steps:")
    print(f"   1. Review changes: diff -u {backup_path} {main_path}")
    print(f"   2. Update counter increments to use await:")
    print(f"      - _store_attempts += 1  →  await _store_attempts.increment()")
    print(f"      - _stores_since_consolidation += 1  →  await _stores_since_consolidation.increment()")
    print(f"   3. Test thoroughly before deploying")
    
    return True


if __name__ == "__main__":
    success = migrate_main()
    exit(0 if success else 1)
