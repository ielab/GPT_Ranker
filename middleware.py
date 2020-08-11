from anserini_retriever import *


def getDocumentContent(CONF, docid):
    retriever = AnseriniRetriever(CONF)
    doc = retriever.getDoc(docid)
    content = doc.raw()
    return content
