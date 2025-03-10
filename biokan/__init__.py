"""
BioKAN - 生体模倣コルモゴロフアーノルドネットワーク
人間の認知プロセスを模倣し、神経科学的に説明可能な深層学習モデル
"""

__version__ = "0.1.0"
__author__ = "BioKAN Team"

from biokan.core import BioKANModel, create_biokan_classifier
from biokan.xai import BioKANExplainer, FeatureAttributionExplainer

__all__ = [
    'BioKANModel',
    'create_biokan_classifier',
    'BioKANExplainer',
    'FeatureAttributionExplainer'
] 