from .nerf_synthetic_generation import *
from .google_scanned_objects_generation import *
from .realestate_generation import *
from .deepvoxels_generation import *
from .real_iconic_noface_generation import *
from .shiny_generation import *
from .ibrnet_collected_generation import *
from .spaces_dataset_generation import *


dataset_dict = {
    "nerf_synthetic": NerfSynthGenerationDataset,
    "spaces": SpacesFreeGenerationDataset,
    "google_scanned_objects": GoogleScannedGenerationDataset,
    "realestate10k": RealEstateGenerationDataset,
    "deepvoxels": DeepVoxelsGenerationDataset,
    "real_iconic_noface": LLFFGenerationDataset,
    "ibrnet_collected": IBRNetCollectedGenerationDataset,
    "shiny": ShinyGenerationDataset,
}
