import unittest
from unittest.mock import patch, MagicMock, call
from main import AutomatedResponseSystem

class TestAutomatedResponseSystem(unittest.TestCase):

    def setUp(self):
        self.ars = AutomatedResponseSystem()

    @patch('main.logging')
    def test_low_threat_response(self, mock_logging):
        self.ars.low_threat_response("Test low threat")
        mock_logging.warning.assert_called_once_with("Low threat: Test low threat")
        self.assertEqual(self.ars.monitoring_level, 2)

    @patch('main.logging')
    def test_medium_threat_response(self, mock_logging):
        self.ars.medium_threat_response("Test medium threat")
        mock_logging.error.assert_called_once_with("Medium threat: Test medium threat")
        self.assertEqual(self.ars.monitoring_level, 2)

    @patch('main.logging')
    def test_high_threat_response(self, mock_logging):
        self.ars.high_threat_response("Test high threat")
        expected_calls = [
            call("High threat: Test high threat"),
            call("Lockdown initiated due to high-level threat: Test high threat"),
            call("All personnel alerted about high-level threat: Test high threat"),
            call("Network access disabled as part of lockdown procedure")
        ]
        mock_logging.critical.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_logging.critical.call_count, 4)
        self.assertEqual(self.ars.monitoring_level, 1)  # Monitoring level shouldn't change for high threats

    def test_increase_monitoring(self):
        initial_level = self.ars.monitoring_level
        self.ars.increase_monitoring()
        self.assertEqual(self.ars.monitoring_level, initial_level + 1)

        # Test max level
        self.ars.monitoring_level = 5
        self.ars.increase_monitoring()
        self.assertEqual(self.ars.monitoring_level, 5)

    @patch('main.logging')
    def test_log_threat_details(self, mock_logging):
        self.ars.log_threat_details("Test threat", "low")
        mock_logging.info.assert_called_once()

    @patch('main.logging')
    def test_alert_security_team(self, mock_logging):
        self.ars.alert_security_team("Test alert")
        mock_logging.info.assert_called_once_with("Security team alerted about threat: Test alert")

    @patch('main.logging')
    def test_initiate_lockdown(self, mock_logging):
        self.ars.initiate_lockdown("Test lockdown")
        mock_logging.critical.assert_called_once_with("Lockdown initiated due to high-level threat: Test lockdown")

    @patch('main.logging')
    def test_alert_all_personnel(self, mock_logging):
        self.ars.alert_all_personnel("Test alert all")
        mock_logging.critical.assert_called_once_with("All personnel alerted about high-level threat: Test alert all")

    @patch('main.logging')
    def test_disable_network_access(self, mock_logging):
        self.ars.disable_network_access()
        mock_logging.critical.assert_called_once_with("Network access disabled as part of lockdown procedure")

if __name__ == '__main__':
    unittest.main()
