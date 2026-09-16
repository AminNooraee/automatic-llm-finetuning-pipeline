import unittest

from fine_tuning_pipeline.dataset_adapters import (
    AdapterContext,
    AdapterRegistry,
    AdapterStage,
    AlpacaAdapter,
    DatasetFormatDetectionError,
)


class DatasetRegistryTests(unittest.TestCase):
    def test_duplicate_names_are_rejected(self):
        registry = AdapterRegistry(AdapterStage.SCHEMA)
        registry.register(AlpacaAdapter())
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(AlpacaAdapter())

    def test_unknown_manual_format_lists_supported_formats(self):
        registry = AdapterRegistry(AdapterStage.SCHEMA)
        registry.register(AlpacaAdapter())
        with self.assertRaises(DatasetFormatDetectionError) as raised:
            registry.select([], "unknown", AdapterContext())
        self.assertIn("Supported formats: auto, alpaca", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
