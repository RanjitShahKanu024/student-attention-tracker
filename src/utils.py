import csv
from datetime import datetime

# ✅ Log blink data to CSV with name and timestamp
def log_blink(csv_writer, total_blinks, name="Unknown"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_writer.writerow([name, timestamp, total_blinks])

# ✅ Setup a CSV file for logging blinks
def setup_csv(filename="blink_log.csv"):
    csv_file = open(filename, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Name", "Timestamp", "Blink Count"])  # Header row
    return csv_file, writer
