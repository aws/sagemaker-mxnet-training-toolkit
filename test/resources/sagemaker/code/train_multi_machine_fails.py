import logging
logger = logging.getLogger(__name__)


def train(current_host, hosts, **kwargs):
    """All hosts except one succeed."""
    hosts = sorted(hosts)
    my_index = hosts.index(current_host)
    if my_index == 0:
        raise Exception("Host zero is failing")
    logger.info("Not host zero, not failing")
