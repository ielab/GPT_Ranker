from pyserini.search import SimpleSearcher


class AnseriniRetriever:

    def __init__(self, CONF):
        self.CONF = CONF
        self.indexPath = CONF["INDEX"]
        self.searcher = SimpleSearcher(self.indexPath)

    def getDoc(self, docid):
        hit = self.searcher.doc(docid)
        return hit
