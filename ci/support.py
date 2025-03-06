import os
import re
import subprocess
import warnings

try:
    from junitparser import Error, Failure, JUnitXml, Skipped
except ModuleNotFoundError:
    warnings.warn("JunitParser is not installed.", stacklevel=1)

ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ReportTests:
    delim = "=" * 120

    def __init__(self, dirpath, header):
        self._header = header

        self.passed = 0
        self.skipped = 0
        self.failed = 0

        self.total_time = 0
        self.summary = {"failed": {}, "skipped": {}}
        self._parse(dirpath)

    def __repr__(self):
        return f"{self._header}(PASSED: {self.passed}, FAILED: {self.failed}, SKIPPED: {self.skipped})"

    def __str__(self):
        # Process to MD
        md = []
        # Brief Summary
        table = (
            f"|Passed :heavy_check_mark:     |Failed :x:|Skipped :warning:        |Time :clock9:     |\n"
            f"|------------------------------|---------------|--------------------|------------------|\n"
            f"|{self.passed}                 |{self.failed}  |{self.skipped}      |{self.total_time} |"
        )
        md.append(table)

        # Failed and Skipped Summary
        for verdict, datas in self.summary.items():
            emoji = ":warning:" if verdict == "skipped" else ":x:"

            verdict_table = (
                f"|TestSuite       |TestCase  {emoji}          |Time :clock9: |\n"
                f"|----------------|---------------------------|--------------|\n"
            )
            verdict_details = ""
            for data in datas:
                verdict_table_entry = (
                    f"|{data.classname}|[{data.name}](#user-content-{data.name.lower()})|{data.time}|\n"
                )
                verdict_data_entry = (
                    f"<details><summary>\n"
                    f'<a name="{data.name}"></a>{data.classname}:{data.name}\n'
                    f"</summary>\n\n"
                    f"```python\n"
                    f"{datas[data]}\n"
                    f"```\n"
                    f"</details>\n"
                )
                verdict_table += verdict_table_entry
                verdict_details += verdict_data_entry
            if datas:
                md.append(
                    f"<details><summary>\n\n"
                    f"#### View {verdict.capitalize()} Summary\n"
                    f"</summary>\n\n"
                    f"{verdict_table}\n"
                    f"</details>\n"
                )
                md.append("___\n")
                md.append(f"#### {verdict.capitalize()} Details:\n")
                md.append(f"{verdict_details}\n")
                md.append("___\n")

        md = "\n".join(md)
        if self._header:
            md = f"{self._header}\n{md}"
        return md

    def _parse(self, dirpath):
        # Parse Directory
        for root, _, files in os.walk(dirpath):
            # Filter only xml files
            for file in filter(lambda is_xml: is_xml.endswith(".xml"), files):
                filepath = os.path.join(root, file)
                try:
                    testsuite = JUnitXml.fromfile(filepath)
                    self.total_time += testsuite.time

                    # If there are multiple testsuites
                    if testsuite._tag == "testsuites":
                        self._parse_testsuites(testsuite)
                    else:
                        self._parse_testcases(testsuite)
                except Exception as error:
                    print(error)

    def _parse_testcases(self, testsuite):
        for testcase in testsuite:
            # Means is PASSED
            if not testcase.result:
                self.passed += 1

            # Collect data for FAILED and SKIPPED tests.
            for tc_result in testcase.result:
                if isinstance(tc_result, Failure) or isinstance(tc_result, Error):
                    self.summary["failed"][testcase] = (
                        f"STDOUT: \n{self.delim}\n{testcase.system_out}\n\n"
                        f"STDERR: \n{self.delim}\n{testcase.system_err}\n\n"
                        f"TEXT: \n{self.delim}\n{tc_result.text}\n\n"
                        f"MESSAGE: \n{self.delim}\n{tc_result.message}\n\n"
                    )
                    self.failed += 1
                elif isinstance(tc_result, Skipped):
                    self.summary["skipped"][testcase] = (
                        f"STDOUT: \n{self.delim}\n{testcase.system_out}\n\n"
                        f"STDERR: \n{self.delim}\n{testcase.system_err}\n\n"
                        f"TEXT: \n{self.delim}\n{tc_result.message}"
                    )
                    self.skipped += 1

    def _parse_testsuites(self, testsuites):
        for testsuite in testsuites:
            self._parse_testcases(testsuite)


def report_coverage():
    header = "## Coverage Report:\n"
    try:
        content = subprocess.check_output("coverage report", encoding="utf-8", cwd=ROOT)
    except:
        return

    # Print Raw Data
    raw_data = f"{header}\n{content}"
    print(raw_data)

    # Process to MD
    new_content = f"```shell\n{content}```"
    data = f"{header}\n{new_content}"

    write_summary(data)


def report_tests(header=""):
    results = os.path.join(ROOT, "test-reports")
    report = ReportTests(dirpath=results, header=header)
    write_summary(str(report))


def report_style():
    header = "## Style Report:\n"
    md = []
    for root, _, files in os.walk(os.path.join(ROOT, ".tox", "style", "log")):
        for file in files:
            filepath = os.path.join(root, file)
            print(f"Checking File: {file}")
            with open(filepath, encoding="cp1252", errors="ignore") as fpr:
                content = fpr.read()
            # Extract Summary
            result = re.search(
                pattern=r"cmd: (black|isort).*?\n([\s\S]+)",
                string=content,
                flags=re.MULTILINE,
            )
            if result:
                # First Group
                check_type = result.group(1)
                output = result.group(2)
                if output.startswith("exit_code: 0"):
                    emoji = ":heavy_check_mark:"
                else:
                    emoji = ":x:"
                md.append(f"### {check_type.capitalize()} Check: {emoji}\n")
                md.append(f"```log\n{output}\n```\n")
                md.append("\n___")

    md = "\n".join(md)
    if header:
        md = f"{header}\n{md}"

    write_summary(md)


def write_summary(content):
    filepath = os.getenv("GITHUB_STEP_SUMMARY", "")
    if not filepath:
        return
    # Append Data
    with open(filepath, "a") as fpa:
        fpa.write(content)


if __name__ == "__main__":
    report_style()
