class IPYNB_GRAMMAR:
    METADATA_KEY = 'metadata'
    CELLS_KEY = 'cells'
    CELL_TYPE_KEY = 'cell_type'
    EXECUTION_COUNT_KEY = 'execution_count'
    OUTPUTS_KEY = 'outputs'
    SOURCE_KEY = 'source'
    NBFORMAT_KEY = 'nbformat'
    NBFORMAT_MINOR_KEY = 'nbformat_minor'

    class DEFAULTS:
        DEFAULT_CELL_METADATA = {}
        DEFAULT_EXECUTION_COUNT = None
        DEFAULT_OUTPUTS = []

        DEFAULT_NB_METADATA = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.7.8"
            }
        }
        DEFAULT_NBFORMAT = 4
        DEFAULT_NBFORMAT_MINOR = 1
