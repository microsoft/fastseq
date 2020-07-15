from absl import logging
import fairseq
from fairseq.sequence_generator import SequenceGenerator

LATEST_VERSION = 'latest'


# TODO: Is there a better way to identify the latest release and the master branch?
def get_fairseq_version():
    version = fairseq.__version__
    if version == '0.9.0' and hasattr(SequenceGenerator, 'finalize_hypos'):
        return LATEST_VERSION
    return version


def apply_fairseq_optimization():
    version = get_fairseq_version()
    if version == '0.9.0':
        import fastseq.optimiser.fairseq.beam_search_optimiser_v1
        logging.info("fairseq == {} has been optimized.".format(version))
        return
    elif version == LATEST_VERSION:
        import fastseq.optimiser.fairseq.beam_search_optimiser_v2
        logging.info("fairseq == {} has been optimized.".format(version))
        return
    else:
        logging.warning(
            "fairseq == {} is not supported yet, please upgrade it to 0.9.0 or above"
            .format(version))
