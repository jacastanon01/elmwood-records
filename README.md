# Project Overview

Since 1872, [Elmwood Cemetery](https://elmwoodcemeterykc.org/) has been a historic burial ground in Kansas City, Missouri and is listed in the National Register of Historic Places. The cemetery is currently undergoing an archival project and need help with digitizing their burial records into a searchable database with the [CryptKeeper Cemetery Software](https://ckonline.tbgtom.com/).

This project is an effort to create a digital archive of all of Elmwood's burial records, utilizing Tesseract OCR to extract text from scanned documents.

## Process 📝

### Authentication and Google Drive Integration

Utilized google-auth, google-auth-oauthlib, and google-api-python-client libraries to authenticate and access files stored in Google Drive.

### PDF to Image Conversion

Employed the pdf2image library to convert PDF pages into image files, facilitating OCR processing.

### Text Extraction with Tesseract OCR

Applied Tesseract OCR through the pytesseract library to extract text from the converted image files, allowing for the digitization of scanned documents.
