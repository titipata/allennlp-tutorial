import json
from typing import Iterator, List, Dict, Optional

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary


@DatasetReader.register("venue_reader")
class PublicationDatasetReader(DatasetReader):
    """
    DatasetReader for publication and venue dataaet
    """
    def __init__(self, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None, 
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        Read publication and venue dataset in JSON format
        
        Data is in the following format:
            {"title": ..., "paperAbstract": ..., "venue": ...}
        """
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                title = paper_json['title']
                abstract = paper_json['paperAbstract']
                venue = paper_json['venue']
                yield self.text_to_instance(title, abstract, venue)
        
    def text_to_instance(self, 
                         title: str, 
                         abstract: str, 
                         venue: str=None) -> Instance:
        """
        Turn title, abstract, and venue to instance
        """
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields = {'title': title_field, 
                  'abstract': abstract_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields)