import os
import subprocess
import pytest
from pathlib import Path
from PIL import Image

def test_spritegrid_cli(tmp_path):
    # Setup paths
    input_image = Path("tests/data/input/centurion.png")
    output_file = tmp_path / "output.png"
    
    # Run CLI command with proper arguments
    result = subprocess.run(
        ["python", "-m", "spritegrid.cli", str(input_image), "-o", str(output_file)],
        capture_output=True,
        text=True
    )
    
    # Verify CLI execution
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    
    # Verify output file was created
    assert output_file.exists(), f"Output file {output_file} was not created"
    
    # Verify output is a valid image
    try:
        img = Image.open(output_file)
        img.verify()  # Verify it's a valid image
        # Check that the image has reasonable dimensions
        assert img.width > 0 and img.height > 0, "Output image has invalid dimensions"
    except Exception as e:
        pytest.fail(f"Output file is not a valid image: {e}")

    # Compare with expected image
    expected_image = Path("tests/data/expected/centurion-default.png")
    assert expected_image.exists(), "Expected image does not exist"

    expected_img = Image.open(expected_image)
    output_img = Image.open(output_file)

    # Compare dimensions
    assert output_img.width == expected_img.width and output_img.height == expected_img.height, "Image dimensions do not match expected"

    # Compare pixel data
    assert list(output_img.getdata()) == list(expected_img.getdata()), "Image content does not match expected"

def test_spritegrid_cli_with_background_removal(tmp_path):
    """Test CLI with background removal before processing."""
    # Setup paths
    input_image = Path("assets/examples/centurion.png")
    output_file = tmp_path / "output.png"
    
    # Run CLI command with background removal
    result = subprocess.run(
        ["python", "-m", "spritegrid.cli", str(input_image), "-o", str(output_file), "-b"],
        capture_output=True,
        text=True
    )
    
    # Verify CLI execution
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    
    # Verify output file was created
    assert output_file.exists(), f"Output file {output_file} was not created"
    
    # Verify output is a valid image
    try:
        img = Image.open(output_file)
        img.verify()
        assert img.width > 0 and img.height > 0
    except Exception as e:
        pytest.fail(f"Output file is not a valid image: {e}")

    # Compare with expected image
    expected_image = Path("tests/data/expected/centurion-b.png")
    assert expected_image.exists(), "Expected image does not exist"

    expected_img = Image.open(expected_image)
    output_img = Image.open(output_file)

    # Compare dimensions
    assert output_img.width == expected_img.width and output_img.height == expected_img.height, "Image dimensions do not match expected"

    # Compare pixel data
    assert list(output_img.getdata()) == list(expected_img.getdata()), "Image content does not match expected"

def test_spritegrid_cli_with_cropping(tmp_path):
    """Test CLI with automatic cropping."""
    # Setup paths
    input_image = Path("tests/data/input/centurion.png")
    output_file = tmp_path / "output.png"
    
    # Run CLI command with cropping
    result = subprocess.run(
        ["python", "-m", "spritegrid.cli", str(input_image), "-o", str(output_file), "-c"],
        capture_output=True,
        text=True
    )
    
    # Verify CLI execution
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    
    # Verify output file was created
    assert output_file.exists(), f"Output file {output_file} was not created"
    
    # Verify output is a valid image with different dimensions
    try:
        img = Image.open(output_file)
        img.verify()
        assert img.width > 0 and img.height > 0
        # Original image dimensions (from test image)
        original_img = Image.open(input_image)
        assert img.width < original_img.width or img.height < original_img.height, "Cropping did not reduce image size"
    except Exception as e:
        pytest.fail(f"Output file is not a valid image: {e}")

    # Compare with expected image
    expected_image = Path("tests/data/expected/centurion.png")
    assert expected_image.exists(), "Expected image does not exist"

    expected_img = Image.open(expected_image)
    output_img = Image.open(output_file)

    # Compare dimensions
    assert output_img.width == expected_img.width and output_img.height == expected_img.height, "Image dimensions do not match expected"

    # Compare pixel data
    assert list(output_img.getdata()) == list(expected_img.getdata()), "Image content does not match expected"

def test_spritegrid_cli_with_ascii_output(tmp_path):
    """Test CLI with ASCII text output."""
    # Setup paths
    input_image = Path("tests/data/input/centurion.png")
    output_file = tmp_path / "output.txt"
    
    # Run CLI command with ASCII output
    result = subprocess.run(
        ["python", "-m", "spritegrid.cli", str(input_image), "-o", str(output_file), "-a", "1"],
        capture_output=True,
        text=True
    )
    
    # Verify CLI execution
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
    
    # Verify output file was created and is a text file
    assert output_file.exists(), f"Output file {output_file} was not created"
    assert output_file.suffix == ".txt", "Output file is not a text file"
    
    # Verify text file content
    with open(output_file, 'r') as f:
        content = f.read()
        assert len(content) > 0, "ASCII output file is empty"

    # Compare with expected text
    expected_text = Path("tests/data/expected/centurion-ansi.txt")
    assert expected_text.exists(), "Expected text file does not exist"

    with open(expected_text, 'r') as f:
        expected_content = f.read()

    assert content == expected_content, "ASCII output does not match expected text"
