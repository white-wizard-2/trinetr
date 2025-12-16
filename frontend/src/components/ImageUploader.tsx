import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import './ImageUploader.css'

interface ImageUploaderProps {
  onImageUpload: (file: File) => void
  imageFile: File | null
}

function ImageUploader({ onImageUpload, imageFile }: ImageUploaderProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onImageUpload(acceptedFiles[0])
    }
  }, [onImageUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    },
    maxFiles: 1
  })

  return (
    <div className="image-uploader">
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        <div className="dropzone-content">
          {isDragActive ? (
            <p>Drop the image here...</p>
          ) : (
            <div className="dropzone-text">
              <p>Drag & drop an image here, or click to select</p>
              <p className="hint">Supports: PNG, JPG, JPEG, GIF, WEBP</p>
            </div>
          )}
          {imageFile && (
            <div className="dropzone-preview">
              <img 
                src={URL.createObjectURL(imageFile)} 
                alt="Preview" 
                className="preview-image-inline"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ImageUploader

