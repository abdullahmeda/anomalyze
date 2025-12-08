"""Explore the raw HuggingFace dataset."""

import logging
from collections import Counter

import pandas as pd
from datasets import load_dataset

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")['train']
    df = dataset.to_pandas()

    logger.info("=" * 60)
    logger.info("DATASET OVERVIEW")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")

    logger.info("\n" + "=" * 60)
    logger.info("CATEGORICAL COLUMN DISTRIBUTIONS")
    logger.info("=" * 60)

    # Type distribution
    logger.info("\n--- Ticket Type ---")
    logger.info(df['type'].value_counts(dropna=False))

    # Queue distribution
    logger.info("\n--- Queue ---")
    logger.info(df['queue'].value_counts(dropna=False))

    # Priority distribution
    logger.info("\n--- Priority ---")
    logger.info(df['priority'].value_counts(dropna=False))

    # Language distribution
    logger.info("\n--- Language ---")
    logger.info(df['language'].value_counts(dropna=False))

    logger.info("\n" + "=" * 60)
    logger.info("TEXT FIELD STATISTICS")
    logger.info("=" * 60)

    # Text length analysis
    df['subject_len'] = df['subject'].fillna('').str.len()
    df['body_len'] = df['body'].fillna('').str.len()
    df['answer_len'] = df['answer'].fillna('').str.len()

    logger.info("\n--- Subject Length ---")
    logger.info(df['subject_len'].describe())

    logger.info("\n--- Body Length ---")
    logger.info(df['body_len'].describe())

    logger.info("\n--- Answer Length ---")
    logger.info(df['answer_len'].describe())

    logger.info("\n" + "=" * 60)
    logger.info("TAG ANALYSIS")
    logger.info("=" * 60)

    # Count how many tags each ticket has
    tag_cols = ['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8']
    df['num_tags'] = df[tag_cols].notna().sum(axis=1)
    logger.info("\n--- Number of Tags per Ticket ---")
    logger.info(df['num_tags'].value_counts().sort_index())

    # Most common tags overall
    all_tags = []
    for col in tag_cols:
        all_tags.extend(df[col].dropna().tolist())

    tag_counts = Counter(all_tags)
    logger.info("\n--- Top 20 Most Common Tags ---")
    for tag, count in tag_counts.most_common(20):
        logger.info(f"  {tag}: {count}")

    logger.info("\n" + "=" * 60)
    logger.info("MISSING VALUES SUMMARY")
    logger.info("=" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'missing': missing, 'percent': missing_pct})
    logger.info(missing_df[missing_df['missing'] > 0])

    logger.info("\n" + "=" * 60)
    logger.info("VERSION FIELD ANALYSIS")
    logger.info("=" * 60)
    logger.info(df['version'].describe())
    logger.info(f"\nUnique versions: {df['version'].dropna().nunique()}")

    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE TICKETS BY TYPE")
    logger.info("=" * 60)
    for ticket_type in df['type'].dropna().unique()[:5]:
        logger.info(f"\n--- Sample {ticket_type} ---")
        sample = df[df['type'] == ticket_type].iloc[0]
        logger.info(f"Subject: {sample['subject'][:80] if pd.notna(sample['subject']) else 'N/A'}...")
        logger.info(f"Queue: {sample['queue']}, Priority: {sample['priority']}, Language: {sample['language']}")
        logger.info(f"Tags: {[sample[t] for t in tag_cols if pd.notna(sample[t])]}")


if __name__ == "__main__":
    main()
