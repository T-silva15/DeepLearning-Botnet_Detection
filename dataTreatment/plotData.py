import matplotlib.pyplot as plt

datasets = ['CIC-IOT-2023', 'Kitsune', 'N-BaloT', 'IoT-IDS', 'CIC-IDS2018', 'NSL-KDD']
total_records = [45588384, 27170754, 7062606, 1191264, 16232943, 4898431]
mirai_records = [2598124, 764137, 3668437, 67193, 0, 0]
iot_devices = [105, 21, 9, 0, 0, 0]

# 1. Number of IoT Devices Comparison
plt.figure(figsize=(10, 6))
plt.bar(datasets, iot_devices, color='skyblue')
plt.xlabel('Datasets')
plt.ylabel('Number of IoT Devices')
plt.title('IoT Devices in Datasets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Total Records Comparison
plt.figure(figsize=(10, 6))
plt.bar(datasets, total_records, color='lightgreen')
plt.xlabel('Datasets')
plt.ylabel('Total Records')  # Modified y-label
plt.title('Total Records in Datasets')
plt.xticks(rotation=45, ha='right')

# Format y-axis to show values in thousands
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

plt.tight_layout()
plt.show()

# 3. Mirai Records Comparison
plt.figure(figsize=(10, 6))
plt.bar(datasets, mirai_records, color='salmon')
plt.xlabel('Datasets')
plt.ylabel('Mirai Records')  # Modified y-label
plt.title('Mirai Records in Datasets')
plt.xticks(rotation=45, ha='right')

# Format y-axis to show values in thousands
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

plt.tight_layout()
plt.show()

# # 4. Relationship between IoT Devices and Mirai Records
# plt.figure(figsize=(8, 6))
# plt.scatter(iot_devices, mirai_records, color='purple')
# plt.xlabel('Number of IoT Devices')
# plt.ylabel('Mirai Records')
# plt.title('IoT Devices vs. Mirai Records')
# plt.grid(True)
# plt.tight_layout()
# plt.show()