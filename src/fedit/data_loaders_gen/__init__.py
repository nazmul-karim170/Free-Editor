from .nerf_synthetic_generation import *
from .google_scanned_objects_generation import *
from .realestate_generation import *
from .deepvoxels_generation import *
from .real_iconic_noface_generation import *
from .shiny_generation import *
from .ibrnet_collected_generation import *
from .spaces_dataset_generation import *


dataset_dict = {
    "nerf_synthetic_generation": NerfSynthGenerationDataset,
    "spaces_generation": SpacesFreeGenerationDataset,
    "google_scanned_object_generation": GoogleScannedGenerationDataset,
    "realestate_generation": RealEstateGenerationDataset,
    "deepvoxels_generation": DeepVoxelsGenerationDataset,
    "real_iconic_noface_geneartion": LLFFGenerationDataset,
    "ibrnet_collected_generation": IBRNetCollectedGenerationDataset,
    "shiny_generation": ShinyGenerationDataset,
}
