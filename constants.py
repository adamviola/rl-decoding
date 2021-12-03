BEST_MODEL = ""

DATA_PREFIX = "data/"

OUTPUTS_PREFIX = "outputs/"

DQL_DATA_PREFIX = "train_data/"

TEST_PAIRS = [("newsdiscusstest2015-enfr-src.en.sgm", "newsdiscusstest2015-enfr-ref.fr.sgm")]

TRAIN_PAIRS = [
                ("news-test2008-src.en.sgm", "news-test2008-ref.fr.sgm"),
                ("newstest2009-src.en.sgm", "newstest2009-ref.fr.sgm"),
                ("newstest2010-src.en.sgm", "newstest2010-ref.fr.sgm"),
                ("newstest2011-src.en.sgm", "newstest2011-ref.fr.sgm"),
                # ("newstest2012-src.en.sgm", "newstest2012-ref.fr.sgm"),
                ("newstest2013-src.en.sgm", "newstest2013-ref.fr.sgm"),
                ("newstest2014-fren-src.en.sgm", "newstest2014-fren-ref.fr.sgm")
                ]

VAL_PAIRS = [("newsdiscussdev2015-enfr-src.en.sgm", "newsdiscussdev2015-enfr-ref.fr.sgm")]