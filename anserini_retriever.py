from pyserini.search import SimpleSearcher

"""
THIS FILE MAINLY CONTAINS ANSERINI RELATED FUNCTIONS
"""


class AnseriniRetriever:

    def __init__(self, CONF):
        self.CONF = CONF
        self.searcher = SimpleSearcher(CONF["INDEX"])

    def getDoc(self, docid):
        hit = self.searcher.doc(docid)
        return hit
