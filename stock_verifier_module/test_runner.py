"""
Test Runner for Stock Verifier

This script runs test cases defined in test_cases.json and generates detailed reports.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from stock_verifier_improved import (
    verify_and_correct_stock,
    SearchStrategy,
    StockCorrectionResult,
    get_vector_store,
    codes_match,
)


# ============================================================================
# Test Result Classes
# ============================================================================

class TestStatus(Enum):
    """Test execution status"""
    PASSED = "[PASSED]"
    FAILED = "[FAILED]"
    SKIPPED = "[SKIPPED]"
    ERROR = "[ERROR]"


@dataclass
class TestResult:
    """Result of a single test case"""
    test_id: str
    description: str
    status: TestStatus
    input_data: Dict[str, Any]
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    correction_result: Optional[StockCorrectionResult]
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        if self.correction_result:
            result['correction_result'] = {
                'original_stock_name': self.correction_result.original_stock_name,
                'original_stock_code': self.correction_result.original_stock_code,
                'corrected_stock_name': self.correction_result.corrected_stock_name,
                'corrected_stock_code': self.correction_result.corrected_stock_code,
                'confidence': self.correction_result.confidence,
                'correction_applied': self.correction_result.correction_applied,
                'confidence_level': self.correction_result.confidence_level,
                'reasoning': self.correction_result.reasoning,
                'search_strategy': self.correction_result.search_strategy,
            }
        return result


@dataclass
class TestSummary:
    """Summary of test execution"""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    execution_time_ms: float = 0.0
    pass_rate: float = 0.0
    
    def calculate_pass_rate(self):
        """Calculate pass rate percentage"""
        if self.total_tests > 0:
            self.pass_rate = (self.passed / self.total_tests) * 100.0
        else:
            self.pass_rate = 0.0


# ============================================================================
# Test Loader
# ============================================================================

class TestLoader:
    """Loads test cases from JSON file"""
    
    def __init__(self, test_file: str = "test_cases.json"):
        self.test_file = Path(test_file)
        self.test_data = None
        self.test_cases = []
        self.config = {}
    
    def load(self) -> bool:
        """Load test cases from file"""
        try:
            if not self.test_file.exists():
                logging.error(f"Test file not found: {self.test_file}")
                return False
            
            with open(self.test_file, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
            
            self.test_cases = self.test_data.get('test_cases', [])
            self.config = self.test_data.get('test_configuration', {})
            
            logging.info(f"Loaded {len(self.test_cases)} test cases from {self.test_file}")
            return True
        
        except Exception as e:
            logging.error(f"Failed to load test file: {str(e)}")
            return False
    
    def get_test_cases(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get test cases, optionally filtered by tags.
        
        Args:
            tags: List of tags to filter by (OR logic). None returns all.
        
        Returns:
            List of test case dictionaries
        """
        if not tags:
            return self.test_cases
        
        filtered = []
        for test_case in self.test_cases:
            test_tags = test_case.get('tags', [])
            if any(tag in test_tags for tag in tags):
                filtered.append(test_case)
        
        return filtered


# ============================================================================
# Test Executor
# ============================================================================

class TestExecutor:
    """Executes test cases and validates results"""
    
    def __init__(self, strategy: SearchStrategy = SearchStrategy.OPTIMIZED):
        self.strategy = strategy
        self.vector_store = None
        self.results: List[TestResult] = []
    
    def initialize(self) -> bool:
        """Initialize vector store connection"""
        try:
            self.vector_store = get_vector_store()
            if not self.vector_store.is_available:
                return self.vector_store.initialize()
            return True
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {str(e)}")
            return False
    
    def execute_test_case(self, test_case: Dict[str, Any]) -> TestResult:
        """Execute a single test case"""
        test_id = test_case.get('id', 'UNKNOWN')
        description = test_case.get('description', '')
        input_data = test_case.get('input', {})
        expected = test_case.get('expected', {})
        tags = test_case.get('tags', [])
        
        logging.info(f"Executing test: {test_id} - {description}")
        
        # Start timing
        start_time = datetime.now()
        
        try:
            # Execute verification
            result = verify_and_correct_stock(
                stock_name=input_data.get('stock_name'),
                stock_code=input_data.get('stock_code'),
                vector_store=self.vector_store,
                strategy=self.strategy,
            )
            
            # Determine actual results
            actual_code = result.corrected_stock_code or result.original_stock_code
            actual_name = result.corrected_stock_name or result.original_stock_name
            
            actual = {
                'stock_code': actual_code,
                'stock_name': actual_name,
            }
            
            # Validate against expected
            status = self._validate_result(expected, actual)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            test_result = TestResult(
                test_id=test_id,
                description=description,
                status=status,
                input_data=input_data,
                expected=expected,
                actual=actual,
                correction_result=result,
                execution_time_ms=execution_time,
                tags=tags,
            )
            
            return test_result
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            test_result = TestResult(
                test_id=test_id,
                description=description,
                status=TestStatus.ERROR,
                input_data=input_data,
                expected=expected,
                actual={},
                correction_result=None,
                error_message=str(e),
                execution_time_ms=execution_time,
                tags=tags,
            )
            
            logging.error(f"Test {test_id} failed with error: {str(e)}")
            return test_result
    
    def _validate_result(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> TestStatus:
        """Validate actual results against expected"""
        # Check stock code
        expected_code = expected.get('stock_code')
        actual_code = actual.get('stock_code')
        
        # Use codes_match to handle both padded and non-padded versions (e.g., '9992' == '09992')
        if expected_code and not codes_match(actual_code, expected_code):
            return TestStatus.FAILED
        
        # Check stock name
        expected_name = expected.get('stock_name')
        actual_name = actual.get('stock_name')
        
        if expected_name and actual_name != expected_name:
            return TestStatus.FAILED
        
        return TestStatus.PASSED
    
    def run_all_tests(
        self,
        test_cases: List[Dict[str, Any]],
        stop_on_failure: bool = False
    ) -> List[TestResult]:
        """
        Run all test cases.
        
        Args:
            test_cases: List of test case dictionaries
            stop_on_failure: Stop execution on first failure
        
        Returns:
            List of TestResult objects
        """
        self.results = []
        
        for test_case in test_cases:
            result = self.execute_test_case(test_case)
            self.results.append(result)
            
            if stop_on_failure and result.status == TestStatus.FAILED:
                logging.warning(f"Stopping execution due to failure in test: {result.test_id}")
                break
        
        return self.results


# ============================================================================
# Test Reporter
# ============================================================================

class TestReporter:
    """Generates test reports"""
    
    def __init__(self, results: List[TestResult]):
        self.results = results
        self.summary = TestSummary()
        self._calculate_summary()
    
    def _calculate_summary(self):
        """Calculate test execution summary"""
        self.summary.total_tests = len(self.results)
        
        for result in self.results:
            if result.status == TestStatus.PASSED:
                self.summary.passed += 1
            elif result.status == TestStatus.FAILED:
                self.summary.failed += 1
            elif result.status == TestStatus.SKIPPED:
                self.summary.skipped += 1
            elif result.status == TestStatus.ERROR:
                self.summary.errors += 1
            
            self.summary.execution_time_ms += result.execution_time_ms
        
        self.summary.calculate_pass_rate()
    
    def print_console_report(self):
        """Print detailed report to console"""
        import sys
        import io
        
        # Set UTF-8 encoding for Windows console
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        
        print("\n" + "=" * 80)
        print("STOCK VERIFIER TEST REPORT")
        print("=" * 80)
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tests: {self.summary.total_tests}")
        print(f"Strategy: {self.results[0].correction_result.search_strategy if self.results else 'N/A'}")
        print("=" * 80)
        
        # Print individual test results
        print("\nTEST RESULTS:")
        print("-" * 80)
        
        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. {result.status.value} {result.test_id}: {result.description}")
            print(f"   Tags: {', '.join(result.tags)}")
            print(f"   Input:    Code={result.input_data.get('stock_code')}, Name={result.input_data.get('stock_name')}")
            print(f"   Expected: Code={result.expected.get('stock_code')}, Name={result.expected.get('stock_name')}")
            print(f"   Actual:   Code={result.actual.get('stock_code')}, Name={result.actual.get('stock_name')}")
            
            if result.correction_result:
                print(f"   Confidence: {result.correction_result.confidence:.2%} ({result.correction_result.confidence_level})")
                print(f"   Reasoning: {result.correction_result.reasoning}")
            
            if result.error_message:
                print(f"   Error: {result.error_message}")
            
            print(f"   Execution Time: {result.execution_time_ms:.2f} ms")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests:     {self.summary.total_tests}")
        print(f"Passed:          {self.summary.passed}")
        print(f"Failed:          {self.summary.failed}")
        print(f"Errors:          {self.summary.errors}")
        print(f"Skipped:         {self.summary.skipped}")
        print(f"Pass Rate:       {self.summary.pass_rate:.2f}%")
        print(f"Total Time:      {self.summary.execution_time_ms:.2f} ms")
        print("=" * 80)
        
        # Print failed tests summary
        if self.summary.failed > 0 or self.summary.errors > 0:
            print("\nFAILED/ERROR TESTS:")
            print("-" * 80)
            for result in self.results:
                if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    print(f"  • {result.test_id}: {result.description}")
                    if result.status == TestStatus.FAILED:
                        print(f"    Expected: {result.expected}")
                        print(f"    Actual:   {result.actual}")
                    if result.error_message:
                        print(f"    Error: {result.error_message}")
        
        print()
    
    def save_json_report(self, output_file: str = "test_report.json"):
        """Save detailed report as JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': self.summary.total_tests,
                'passed': self.summary.passed,
                'failed': self.summary.failed,
                'errors': self.summary.errors,
                'skipped': self.summary.skipped,
                'pass_rate': self.summary.pass_rate,
                'execution_time_ms': self.summary.execution_time_ms,
            },
            'results': [result.to_dict() for result in self.results],
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"JSON report saved to: {output_file}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main test execution function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run stock verifier tests')
    parser.add_argument(
        '--strategy',
        choices=['optimized', 'semantic_only', 'exact_only'],
        default='optimized',
        help='Search strategy to use (default: optimized)'
    )
    parser.add_argument(
        '--test-file',
        default='test_cases.json',
        help='Test cases file (default: test_cases.json)'
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        help='Filter tests by tags (OR logic)'
    )
    parser.add_argument(
        '--stop-on-failure',
        action='store_true',
        help='Stop execution on first failure'
    )
    parser.add_argument(
        '--output',
        default='test_report.json',
        help='Output JSON report file (default: test_report.json)'
    )
    
    args = parser.parse_args()
    
    # Map strategy string to enum
    strategy_map = {
        'optimized': SearchStrategy.OPTIMIZED,
        'semantic_only': SearchStrategy.SEMANTIC_ONLY,
        'exact_only': SearchStrategy.EXACT_ONLY,
    }
    strategy = strategy_map[args.strategy]
    
    print("=" * 80)
    print("STOCK VERIFIER TEST RUNNER")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Test File: {args.test_file}")
    if args.tags:
        print(f"Filtered by tags: {', '.join(args.tags)}")
    print("=" * 80)
    
    # Load test cases
    loader = TestLoader(args.test_file)
    if not loader.load():
        print("Failed to load test cases")
        sys.exit(1)
    
    test_cases = loader.get_test_cases(tags=args.tags)
    
    if not test_cases:
        print("Warning: No test cases to run")
        sys.exit(0)
    
    print(f"\nLoaded {len(test_cases)} test cases")
    
    # Initialize executor
    executor = TestExecutor(strategy=strategy)
    
    print("Initializing vector store...")
    if not executor.initialize():
        print("Failed to initialize vector store")
        sys.exit(1)
    
    print("Vector store initialized successfully")
    
    # Run tests
    print(f"\nRunning {len(test_cases)} tests...\n")
    results = executor.run_all_tests(test_cases, stop_on_failure=args.stop_on_failure)
    
    # Generate report
    reporter = TestReporter(results)
    reporter.print_console_report()
    reporter.save_json_report(args.output)
    
    # Exit with appropriate code
    if reporter.summary.failed > 0 or reporter.summary.errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

