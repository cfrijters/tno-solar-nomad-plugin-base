
import pandas as pd
from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser


class BatchParser(MatchingParser):

    def parse(self, mainfile: str, archive: EntryArchive, logger):
        pass
        #upload_id = archive.metadata.upload_id
        #df = pd.read_excel(mainfile, header=[0, 1])
