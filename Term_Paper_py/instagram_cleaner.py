"""
Instagram Scraper Data Cleanup Script
--------------------------------------
Cleans raw Apify Instagram scraper CSV outputs for D2C brand analysis.

Usage:
    python instagram_cleaner.py input.csv output.csv

Author: Research Assistant
Date: Feb 2026
"""

import pandas as pd
import sys
from pathlib import Path


# TARGET BRAND LIST (update as needed)
TARGET_BRANDS = [
    'suta_bombay', 'thewholetruthfoods', 'nua_woman', 'thesouledstore', 
    'plumgoodness', 'mokobara', 'sironahygiene', 'myblissclub',
    'juicy_chemistry', 'jimmysbeverages', 'foxtaleskin', 'wakefitco',
    'indya', 'bombayshirts', 'mcaffeineofficial', 'beminimalist__',
    'bombayshavingcompany', 'insightcosmetic', 'sleepyowlcoffee', 'bira91beer'
]

# COLUMNS TO KEEP (analysis-relevant only)
KEEP_COLS = [
    'id', 'shortCode', 'ownerUsername', 'ownerFullName', 'ownerId',
    'caption', 'type', 'productType',
    'likesCount', 'commentsCount', 'videoViewCount', 'videoPlayCount', 'videoDuration',
    'timestamp',
    'dimensionsHeight', 'dimensionsWidth',
    'isCommentsDisabled', 'isPinned', 'paidPartnership',
    'locationName', 'firstComment',
    'hashtags/0', 'hashtags/1', 'hashtags/2', 'hashtags/3', 'hashtags/4',
    'hashtags/5', 'hashtags/6', 'hashtags/7', 'hashtags/8', 'hashtags/9',
    'mentions/0', 'mentions/1', 'mentions/2', 'mentions/3', 'mentions/4',
    'inputUrl', 'url',
    'musicInfo/artist_name', 'musicInfo/song_name', 'musicInfo/uses_original_audio',
]


def clean_instagram_data(input_path, output_path=None, target_brands=None, verbose=True):
    """
    Clean Instagram scraper CSV output.
    
    Args:
        input_path (str): Path to raw CSV file
        output_path (str): Path to save cleaned CSV (optional)
        target_brands (list): List of target brand usernames (default: TARGET_BRANDS)
        verbose (bool): Print progress messages
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if target_brands is None:
        target_brands = TARGET_BRANDS
    
    if verbose:
        print(f"📂 Loading: {input_path}")
    
    # Load raw data
    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)
    initial_cols = df.shape[1]
    
    if verbose:
        print(f"   Raw data: {initial_rows:,} rows × {initial_cols:,} cols")
    
    # STEP 1: Filter to target brands only
    if 'ownerUsername' not in df.columns:
        raise ValueError("Column 'ownerUsername' not found in input CSV")
    
    df = df[df['ownerUsername'].isin(target_brands)].copy()
    after_filter = len(df)
    
    if verbose:
        print(f"\n🎯 Step 1: Filter to target brands")
        print(f"   Kept: {after_filter:,} / {initial_rows:,} rows ({after_filter/initial_rows*100:.1f}%)")
        print(f"   Removed: {initial_rows - after_filter:,} noise/tagged posts")
    
    # STEP 2: Deduplicate on shortCode
    before_dedup = len(df)
    df = df.drop_duplicates(subset='shortCode', keep='first')
    after_dedup = len(df)
    
    if verbose:
        print(f"\n🔄 Step 2: Deduplicate posts")
        print(f"   Kept: {after_dedup:,} unique posts")
        print(f"   Removed: {before_dedup - after_dedup:,} duplicates")
    
    # STEP 3: Remove rows with negative likes (scraper errors)
    before_neg = len(df)
    df = df[df['likesCount'] >= 0].copy()
    after_neg = len(df)
    
    if verbose:
        print(f"\n🧹 Step 3: Remove negative likes")
        print(f"   Removed: {before_neg - after_neg:,} rows with likesCount < 0")
    
    # STEP 4: Keep only relevant columns
    keep_cols_present = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep_cols_present].copy()
    
    if verbose:
        print(f"\n📊 Step 4: Drop irrelevant columns")
        print(f"   Kept: {len(keep_cols_present)} / {initial_cols:,} columns")
        print(f"   Dropped: {initial_cols - len(keep_cols_present):,} columns")
    
    # STEP 5: Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # STEP 6: Quality checks
    if verbose:
        print(f"\n✅ Final dataset: {len(df):,} rows × {len(keep_cols_present)} cols")
        print(f"\n📈 Brand Breakdown:")
        brand_counts = df['ownerUsername'].value_counts()
        for brand in target_brands:
            count = brand_counts.get(brand, 0)
            status = "✓" if count >= 40 else "⚠" if count >= 10 else "✗"
            print(f"   {status} {brand:25s} {count:4d} posts")
        
        # Missing brands
        missing = [b for b in target_brands if brand_counts.get(b, 0) == 0]
        if missing:
            print(f"\n⚠️  Brands with 0 posts: {', '.join(missing)}")
        
        # Data completeness
        print(f"\n🔍 Key Field Completeness:")
        critical_fields = ['caption', 'likesCount', 'commentsCount', 'timestamp', 'type']
        for field in critical_fields:
            if field in df.columns:
                completeness = (1 - df[field].isna().mean()) * 100
                print(f"   {field:20s} {completeness:5.1f}%")
        
        # Date range
        if 'timestamp' in df.columns:
            print(f"\n📅 Date Range:")
            print(f"   Earliest: {df['timestamp'].min()}")
            print(f"   Latest:   {df['timestamp'].max()}")
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n💾 Saved to: {output_path}")
    
    return df


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python instagram_cleaner.py <input.csv> [output.csv]")
        print("\nExample:")
        print("  python instagram_cleaner.py raw_data.csv clean_data.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # Run cleaning
    df = clean_instagram_data(input_path, output_path, verbose=True)
    
    print("\n" + "="*60)
    print("✅ Cleanup complete!")
    print("="*60)


if __name__ == "__main__":
    main()
