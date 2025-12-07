from datasets import load_dataset
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def main():
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")['train']
    df = dataset.to_pandas()
    
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n" + "=" * 60)
    print("CATEGORICAL COLUMN DISTRIBUTIONS")
    print("=" * 60)
    
    # Type distribution
    print("\n--- Ticket Type ---")
    print(df['type'].value_counts(dropna=False))
    
    # Queue distribution
    print("\n--- Queue ---")
    print(df['queue'].value_counts(dropna=False))
    
    # Priority distribution
    print("\n--- Priority ---")
    print(df['priority'].value_counts(dropna=False))
    
    # Language distribution
    print("\n--- Language ---")
    print(df['language'].value_counts(dropna=False))
    
    print("\n" + "=" * 60)
    print("TEXT FIELD STATISTICS")
    print("=" * 60)
    
    # Text length analysis
    df['subject_len'] = df['subject'].fillna('').str.len()
    df['body_len'] = df['body'].fillna('').str.len()
    df['answer_len'] = df['answer'].fillna('').str.len()
    
    print("\n--- Subject Length ---")
    print(df['subject_len'].describe())
    
    print("\n--- Body Length ---")
    print(df['body_len'].describe())
    
    print("\n--- Answer Length ---")
    print(df['answer_len'].describe())
    
    print("\n" + "=" * 60)
    print("TAG ANALYSIS")
    print("=" * 60)
    
    # Count how many tags each ticket has
    tag_cols = ['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8']
    df['num_tags'] = df[tag_cols].notna().sum(axis=1)
    print("\n--- Number of Tags per Ticket ---")
    print(df['num_tags'].value_counts().sort_index())
    
    # Most common tags overall
    all_tags = []
    for col in tag_cols:
        all_tags.extend(df[col].dropna().tolist())
    
    from collections import Counter
    tag_counts = Counter(all_tags)
    print("\n--- Top 20 Most Common Tags ---")
    for tag, count in tag_counts.most_common(20):
        print(f"  {tag}: {count}")
    
    print("\n" + "=" * 60)
    print("MISSING VALUES SUMMARY")
    print("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'missing': missing, 'percent': missing_pct})
    print(missing_df[missing_df['missing'] > 0])
    
    print("\n" + "=" * 60)
    print("VERSION FIELD ANALYSIS")
    print("=" * 60)
    print(df['version'].describe())
    print(f"\nUnique versions: {df['version'].dropna().nunique()}")
    
    print("\n" + "=" * 60)
    print("SAMPLE TICKETS BY TYPE")
    print("=" * 60)
    for ticket_type in df['type'].dropna().unique()[:5]:
        print(f"\n--- Sample {ticket_type} ---")
        sample = df[df['type'] == ticket_type].iloc[0]
        print(f"Subject: {sample['subject'][:80] if pd.notna(sample['subject']) else 'N/A'}...")
        print(f"Queue: {sample['queue']}, Priority: {sample['priority']}, Language: {sample['language']}")
        print(f"Tags: {[sample[t] for t in tag_cols if pd.notna(sample[t])]}")


if __name__ == "__main__":
    main()
