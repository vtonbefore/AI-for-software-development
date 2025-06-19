import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
from datetime import datetime

class LoginTestFramework(unittest.TestCase):
    """
    AI-Assisted Automated Testing for Login Functionality
    This framework captures success/failure rates and compares different test approaches
    """
    
    def setUp(self):
        """Initialize the web driver and test data"""
        self.driver = webdriver.Chrome()  # Make sure you have chromedriver installed
        self.driver.maximize_window()
        self.base_url = "https://the-internet.herokuapp.com/login"  # Example login page
        self.wait = WebDriverWait(self.driver, 10)
        
        # Test results tracking
        self.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'test_details': []
        }
        
        # AI-suggested test data combinations
        self.test_data = {
            'valid_credentials': [
                {'username': 'tomsmith', 'password': 'SuperSecretPassword!'},
            ],
            'invalid_credentials': [
                {'username': '', 'password': ''},  # Empty fields
                {'username': 'invalid_user', 'password': 'wrong_password'},  # Wrong credentials
                {'username': 'tomsmith', 'password': 'wrong_password'},  # Valid user, wrong password
                {'username': 'invalid_user', 'password': 'SuperSecretPassword!'},  # Invalid user, correct password
                {'username': 'admin\'; DROP TABLE users; --', 'password': 'password'},  # SQL injection attempt
                {'username': 'a' * 256, 'password': 'password'},  # Long username
                {'username': 'user', 'password': 'a' * 256},  # Long password
            ]
        }

    def login_attempt(self, username, password):
        """Helper method to perform login attempt"""
        try:
            self.driver.get(self.base_url)
            
            # Find and fill username field
            username_field = self.wait.until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            username_field.clear()
            username_field.send_keys(username)
            
            # Find and fill password field
            password_field = self.driver.find_element(By.ID, "password")
            password_field.clear()
            password_field.send_keys(password)
            
            # Click login button
            login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            login_button.click()
            
            # Wait for page to load and check result
            time.sleep(2)
            
            # Check for success or failure indicators
            current_url = self.driver.current_url
            page_source = self.driver.page_source
            
            if "secure" in current_url or "You logged into a secure area!" in page_source:
                return True, "Login successful"
            elif "Your username is invalid!" in page_source or "Your password is invalid!" in page_source:
                return False, "Invalid credentials"
            else:
                return False, "Unknown error"
                
        except TimeoutException:
            return False, "Page load timeout"
        except Exception as e:
            return False, f"Exception occurred: {str(e)}"

    def record_test_result(self, test_name, expected_result, actual_result, details):
        """Record test results for analysis"""
        self.test_results['total_tests'] += 1
        
        passed = (expected_result == actual_result)
        if passed:
            self.test_results['passed'] += 1
        else:
            self.test_results['failed'] += 1
            
        self.test_results['test_details'].append({
            'test_name': test_name,
            'expected': expected_result,
            'actual': actual_result,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })

    def test_valid_login_scenarios(self):
        """Test valid login credentials - AI suggested comprehensive approach"""
        print("Testing Valid Login Scenarios...")
        
        for i, creds in enumerate(self.test_data['valid_credentials']):
            test_name = f"Valid Login Test {i+1}"
            success, message = self.login_attempt(creds['username'], creds['password'])
            
            self.record_test_result(test_name, True, success, message)
            self.assertTrue(success, f"Valid login failed: {message}")
            
            # Logout if successful
            if success:
                try:
                    logout_link = self.driver.find_element(By.LINK_TEXT, "Logout")
                    logout_link.click()
                    time.sleep(1)
                except:
                    pass

    def test_invalid_login_scenarios(self):
        """Test invalid login credentials - AI generated edge cases"""
        print("Testing Invalid Login Scenarios...")
        
        for i, creds in enumerate(self.test_data['invalid_credentials']):
            test_name = f"Invalid Login Test {i+1}: {creds.get('description', 'Generic invalid test')}"
            success, message = self.login_attempt(creds['username'], creds['password'])
            
            self.record_test_result(test_name, False, success, message)
            # For invalid credentials, we expect login to fail (success = False)
            self.assertFalse(success, f"Invalid login unexpectedly succeeded with: {creds}")

    def test_ui_elements_presence(self):
        """AI-suggested UI validation tests"""
        print("Testing UI Elements...")
        
        self.driver.get(self.base_url)
        
        # Check for required elements
        required_elements = [
            (By.ID, "username", "Username field"),
            (By.ID, "password", "Password field"),
            (By.CSS_SELECTOR, "button[type='submit']", "Login button")
        ]
        
        for locator_type, locator_value, element_name in required_elements:
            try:
                element = self.wait.until(
                    EC.presence_of_element_located((locator_type, locator_value))
                )
                self.record_test_result(f"UI Test: {element_name}", True, True, "Element found")
                self.assertTrue(element.is_displayed(), f"{element_name} should be visible")
            except:
                self.record_test_result(f"UI Test: {element_name}", True, False, "Element not found")
                self.fail(f"{element_name} not found on page")

    def generate_test_report(self):
        """Generate comprehensive test report with success/failure rates"""
        total = self.test_results['total_tests']
        passed = self.test_results['passed']
        failed = self.test_results['failed']
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        report = {
            'summary': {
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'success_rate': f"{success_rate:.2f}%"
            },
            'ai_advantages_observed': [
                "Generated comprehensive edge cases automatically",
                "Identified security test scenarios (SQL injection)",
                "Suggested boundary value testing (long inputs)",
                "Provided structured test data organization"
            ],
            'detailed_results': self.test_results['test_details']
        }
        
        # Save report to file
        with open('login_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n=== TEST EXECUTION SUMMARY ===")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Detailed report saved to: login_test_report.json")
        
        return report

    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'driver'):
            self.driver.quit()

if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    test_framework = LoginTestFramework()
    
    # Add tests to suite
    test_suite.addTest(LoginTestFramework('test_ui_elements_presence'))
    test_suite.addTest(LoginTestFramework('test_valid_login_scenarios'))
    test_suite.addTest(LoginTestFramework('test_invalid_login_scenarios'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    try:
        print("Starting AI-Assisted Login Testing...")
        runner.run(test_suite)
    finally:
        # Generate final report
        test_framework.generate_test_report()