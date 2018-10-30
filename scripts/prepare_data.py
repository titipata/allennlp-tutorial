"""
Script to prepare training and validation dataset from Medline
"""
import json
import pandas as pd
from allennlp.common.file_utils import cached_path
from sklearn.model_selection import train_test_split

JOURNAL_URL = 'https://raw.githubusercontent.com/titipata/detecting-scientific-claim/master/annotation_tool/journals.txt'

def save_json(ls, file_path):
    """
    Save list of dictionary to JSON
    """
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in ls))


if __name__ == '__main__':
    abstract_df = pd.read_csv('abstract_2010_2018.csv', usecols=['title', 'abstract', 'journal'])
    with open(cached_path(JOURNAL_URL)) as f:
        lines = [l.strip() for l in f.readlines()]
    journal_df = pd.DataFrame(lines, columns=['journal'])
    abstract_df = abstract_df.merge(journal_df)
    abstract_df.rename(columns={'abstract': 'paperAbstract', 'journal': 'venue'}, inplace=True)
    abstract_df.dropna(inplace=True)
    abstract_df['abstract_len'] = abstract_df['paperAbstract'].map(lambda x: len(x.split()))
    abstract_df = abstract_df[abstract_df.abstract_len <= 600]
    abstract_sample_df = abstract_df.sample(n=300000, random_state=400)
    abstract_train_df, abstract_validation = train_test_split(abstract_sample_df, 
                                                            test_size=0.3, 
                                                            stratify=abstract_sample_df['venue'])
    save_json([dict(r) for _, r in abstract_train_df.iterrows()], 'venue_110journals_train.jsonl')
    save_json([dict(r) for _, r in abstract_validation.iterrows()], 'venue_110journals_validation.jsonl')