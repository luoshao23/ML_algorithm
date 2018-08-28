from __future__ import absolute_import, unicode_literals

import pbr.version

from llda.lda import LDA
import llda.utils

__version__ = pbr.version.VersionInfo('lda').version_string()
