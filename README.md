# Flight Data CSV Combiner

A powerful Python tool for combining multiple CSV files from Flight Data Recorder (FDR) extracts into a single, optimized dataset. Supports both legacy CSV formats and modern FDR formats with automatic format detection and intelligent data processing.

## Features

### 🚀 **Smart Format Detection**
- **Auto-detects** CSV format (legacy vs modern FDR format)
- **Handles comment headers** (automatically skips lines starting with `#`)
- **Mixed data type support** (numeric and discrete parameters in same batch)
- **Parameter name extraction** from filename patterns

### 📊 **Data Processing**
- **Timestamp synchronization** using common time base
- **Forward fill** missing values to maintain data continuity
- **Automatic sampling interval detection**
- **Downsampling** support for large datasets
- **Precision control** for numeric values

### 🗜️ **File Size Optimization**
- **Gzip compression** (70-90% size reduction)
- **Data type optimization** (20-50% size reduction)
- **Zero-column removal** (configurable)
- **Memory-efficient processing** with batch support

### 🔧 **Advanced Features**
- **Unicode support** (UTF-8, Latin-1, CP1252 encoding detection)
- **Memory monitoring** and batch processing for large datasets
- **Progress tracking** with detailed status reporting
- **Error handling** with graceful failure recovery

## Supported Formats

### Legacy Format
```csv
Timestamp,Value
0,123.45
125,124.56
250,125.67
```

### Modern FDR Format
```csv
# Flight Data Recorder Extract
# Aircraft: 5H-TCM
# Parameter: Wind Speed FMS 2 (WndSpd2)
# Units: kts
# Data Type: Signed Binary
# Total Sessions: 9
# Total Samples: 6172
Sample,Time(sec),Time(min),Value,Units,SessionNum,SessionID,SessionType
1,0.0,0.00,0.000000,kts,1,FLIGHT_1,FLIGHT
```

## Installation

### Prerequisites
- Python 3.7+
- Required packages:
```bash
pip install pandas numpy tqdm psutil
```

### Clone Repository
```bash
git clone https://github.com/SifuLabs/flight-data-csv-combiner.git
cd flight-data-csv-combiner
```

## Usage

### Basic Usage
```bash
python csv_generator.py
```

The program will prompt you for:
- **Folder path** (contains CSV files to combine)
- **Output filename** (default: `combined_data.csv`)
- **Processing settings** (batch size, precision, optimizations)

### Advanced Configuration

#### File Size Optimization
- **Gzip Compression**: Reduces file size by 70-90%
- **Data Type Optimization**: Automatically uses smaller data types
- **Remove Zero Columns**: Eliminates parameters with all zero values
- **Downsampling**: Reduce temporal resolution (e.g., every 2nd sample)

#### Processing Settings
- **Batch Size**: Number of files processed simultaneously (default: 100)
- **Precision**: Decimal places for numeric values (default: 4)
- **Memory Management**: Automatic optimization for large datasets

## File Naming Conventions

### Modern FDR Format
Parameter names are extracted from filenames using the pattern:
```
5H-TCM_2025_06_19_07_06_22_1024.upk_FLIGHT_9_WndSpd2.csv
                                           ↑
                                    Parameter: WndSpd2
```

Pattern: `*_9_<ParameterName>.csv`

### Data Types
- **Numeric Parameters**: Use `Value` column (e.g., speed, altitude, temperature)
- **Discrete Parameters**: Use `TextValue` column (e.g., boolean states, text values)

## Output Format

The tool generates a unified CSV with:
- **Sample**: Continuous sample index (0, 1, 2, ...)
- **Timestamp**: Common time base (typically in seconds)
- **Parameter Columns**: One column per input file/parameter

### Example Output
```csv
Sample,Timestamp,WndSpd2,WHvyR,AltPres,AirSpd
0,0,0.0,False,35000.0,0.0
1,4,0.0,False,35000.0,0.0
2,8,12.5,False,34995.0,145.2
```

## Performance

### Speed Optimizations
- **Dictionary-based merging** (10-30x faster than DataFrame merges)
- **Vectorized operations** for data type optimization
- **Memory-efficient batch processing**
- **Intelligent timestamp handling**

### Memory Management
- **Automatic batch sizing** for large datasets (500+ files)
- **Garbage collection** between file processing
- **Memory usage monitoring** and reporting

### Benchmark Results
| File Count | Processing Time | Memory Usage |
|------------|----------------|--------------|
| 100 files  | ~30 seconds    | < 1 GB       |
| 500 files  | ~3-5 minutes   | < 2 GB       |
| 1000+ files| ~10-15 minutes | < 4 GB       |

## Error Handling

The tool includes robust error handling for:
- **Missing columns** (graceful skipping with warnings)
- **Encoding issues** (automatic fallback between UTF-8, Latin-1, CP1252)
- **Invalid timestamps** (automatic filtering)
- **Memory constraints** (automatic batch size adjustment)
- **File access errors** (detailed error reporting)

## Configuration Options

### Compression Settings
```python
use_compression=True          # Enable gzip compression
optimize_data_types=True      # Optimize numeric data types
remove_zero_columns=False     # Remove all-zero parameters
downsample_factor=1           # No downsampling (1), 50% (2), 25% (4), etc.
```

### Processing Settings
```python
batch_size=100               # Files per batch
precision=4                  # Decimal places for numeric values
auto_detect_format=True      # Enable automatic format detection
```

## Examples

### Processing Mixed Data Types
```
Input Files:
- 5H-TCM_..._9_WndSpd2.csv    (numeric: wind speed)
- 5H-TCM_..._9_WHvyR.csv      (discrete: wing heavy right)
- 5H-TCM_..._9_AltPres.csv    (numeric: pressure altitude)

Output:
Sample,Timestamp,WndSpd2,WHvyR,AltPres
0,0,0.0,False,35000.0
1,4,12.5,False,34995.0
2,8,15.2,True,34990.0
```

### Large Dataset Processing
For datasets with 1000+ files:
- Automatic batch processing (25 files per batch)
- Memory optimization enabled
- Progress tracking with ETA
- Compressed output recommended

## Troubleshooting

### Common Issues

**Unicode Encoding Errors**
- Tool automatically handles multiple encodings
- Supports UTF-8, Latin-1, and CP1252

**Memory Issues**
- Reduce batch size for large datasets
- Enable compression to reduce output size
- Use downsampling for very large time series

**Performance Issues**
- Use batch processing for 500+ files
- Enable data type optimization
- Consider downsampling for analysis purposes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v2.0.0 (Current)
- ✅ Added FDR format support with comment header handling
- ✅ Automatic parameter name extraction from filenames
- ✅ Mixed numeric/discrete data type support
- ✅ Per-file format detection
- ✅ Enhanced error handling and reporting

### v1.0.0
- ✅ Basic CSV combination functionality
- ✅ Legacy format support
- ✅ File size optimization features
- ✅ Memory-efficient processing

## Author

**Sifu Labs**

## Acknowledgments

- Built for Flight Data Recorder (FDR) analysis workflows
- Optimized for aviation data processing requirements
- Supports industry-standard FDR data formats
