#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📸 Image Preprocessor - Pré-processamento Avançado de Imagens
=============================================================
Auto-rotação EXIF, correção de brilho/contraste, detecção de blur.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter, ExifTags

logger = logging.getLogger(__name__)

# Verificar dependências opcionais
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False


@dataclass
class ImageQualityReport:
    """Relatório de qualidade da imagem."""
    is_blurry: bool
    blur_score: float
    brightness: float
    contrast: float
    sharpness: float
    has_faces: bool
    face_count: int
    dominant_colors: list
    exif_data: dict
    recommendations: list


@dataclass
class PreprocessingResult:
    """Resultado do pré-processamento."""
    image: Image.Image
    was_rotated: bool
    was_enhanced: bool
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    quality_report: Optional[ImageQualityReport]


class ImagePreprocessor:
    """Pré-processador avançado de imagens."""
    
    # Limiar para detecção de blur (Laplacian variance)
    BLUR_THRESHOLD = 100.0
    
    # Limiar para correção de brilho
    BRIGHTNESS_LOW = 0.3
    BRIGHTNESS_HIGH = 0.7
    
    # Resolução mínima para upscale
    UPSCALE_MIN_PIXELS = 640
    
    def __init__(
        self,
        auto_rotate: bool = True,
        auto_enhance: bool = False,
        detect_blur: bool = True,
        detect_faces: bool = False,
        analyze_colors: bool = False,
        deskew: bool = False,
        denoise: bool = False,
        upscale: bool = False,
        binarize: bool = False,
    ):
        self.auto_rotate = auto_rotate
        self.auto_enhance = auto_enhance
        self.detect_blur = detect_blur
        self.detect_faces = detect_faces
        self.analyze_colors = analyze_colors
        self.deskew = deskew
        self.denoise = denoise
        self.upscale = upscale
        self.binarize = binarize
    
    def process(self, image: Image.Image, image_path: Path = None) -> PreprocessingResult:
        """
        Processa uma imagem aplicando todas as transformações configuradas.
        Pipeline: EXIF rotate → deskew → contrast → denoise → upscale → binarize
        """
        original_size = image.size
        was_rotated = False
        was_enhanced = False
        
        # 1. Auto-rotação baseada em EXIF
        if self.auto_rotate:
            image, was_rotated = self._auto_rotate_exif(image)
        
        # 2. Deskew (correção de inclinação)
        if self.deskew and CV2_AVAILABLE and NUMPY_AVAILABLE:
            image = self._deskew_image(image)
        
        # 3. Análise de qualidade
        quality_report = self._analyze_quality(image) if any([
            self.detect_blur, self.detect_faces, self.analyze_colors
        ]) else None
        
        # 4. Auto-enhance (contraste, brilho, nitidez)
        if self.auto_enhance and quality_report:
            image, was_enhanced = self._auto_enhance(image, quality_report)
        
        # 5. Redução de ruído
        if self.denoise and CV2_AVAILABLE and NUMPY_AVAILABLE:
            image = self._denoise_image(image)
        
        # 6. Upscale (se imagem muito pequena)
        if self.upscale:
            image = self._upscale_image(image)
        
        # 7. Binarização (para OCR)
        # Nota: retorna imagem binarizada separada, não altera a principal
        
        return PreprocessingResult(
            image=image,
            was_rotated=was_rotated,
            was_enhanced=was_enhanced,
            original_size=original_size,
            final_size=image.size,
            quality_report=quality_report
        )
    
    def _auto_rotate_exif(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Rotaciona imagem baseado em dados EXIF."""
        try:
            exif = image._getexif()
            if exif is None:
                return image, False
            
            # Encontrar tag de orientação
            orientation_tag = None
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_tag = tag
                    break
            
            if orientation_tag is None or orientation_tag not in exif:
                return image, False
            
            orientation = exif[orientation_tag]
            
            # Aplicar rotação baseada na orientação
            rotations = {
                3: Image.ROTATE_180,
                6: Image.ROTATE_270,
                8: Image.ROTATE_90
            }
            
            if orientation in rotations:
                image = image.transpose(rotations[orientation])
                logger.debug(f"🔄 Imagem rotacionada (EXIF orientation: {orientation})")
                return image, True
            
            # Flip horizontal/vertical
            if orientation == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                return image, True
            elif orientation == 4:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                return image, True
            elif orientation == 5:
                image = image.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
                return image, True
            elif orientation == 7:
                image = image.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
                return image, True
            
            return image, False
            
        except Exception as e:
            logger.debug(f"⚠️ Erro ao processar EXIF: {e}")
            return image, False
    
    def _analyze_quality(self, image: Image.Image) -> ImageQualityReport:
        """Analisa a qualidade da imagem."""
        recommendations = []
        
        # Converter para análise
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calcular blur score
        blur_score = 0.0
        is_blurry = False
        if self.detect_blur and NUMPY_AVAILABLE:
            blur_score = self._calculate_blur_score(image)
            is_blurry = blur_score < self.BLUR_THRESHOLD
            if is_blurry:
                recommendations.append("⚠️ Imagem borrada - considere usar outra foto")
        
        # Calcular brilho
        brightness = self._calculate_brightness(image)
        if brightness < self.BRIGHTNESS_LOW:
            recommendations.append("💡 Imagem muito escura - considere aumentar brilho")
        elif brightness > self.BRIGHTNESS_HIGH:
            recommendations.append("☀️ Imagem muito clara - considere reduzir brilho")
        
        # Calcular contraste
        contrast = self._calculate_contrast(image)
        if contrast < 0.3:
            recommendations.append("🎚️ Baixo contraste - considere aumentar contraste")
        
        # Calcular nitidez
        sharpness = self._calculate_sharpness(image)
        
        # Detectar faces
        has_faces = False
        face_count = 0
        if self.detect_faces and CV2_AVAILABLE:
            face_count = self._detect_faces(image)
            has_faces = face_count > 0
        
        # Analisar cores dominantes
        dominant_colors = []
        if self.analyze_colors and NUMPY_AVAILABLE:
            dominant_colors = self._analyze_dominant_colors(image)
        
        # Extrair EXIF
        exif_data = self._extract_exif(image)
        
        return ImageQualityReport(
            is_blurry=is_blurry,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            has_faces=has_faces,
            face_count=face_count,
            dominant_colors=dominant_colors,
            exif_data=exif_data,
            recommendations=recommendations
        )
    
    def _calculate_blur_score(self, image: Image.Image) -> float:
        """Calcula score de blur usando Laplacian variance."""
        if not NUMPY_AVAILABLE:
            return 0.0
        
        try:
            # Converter para grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            if CV2_AVAILABLE:
                # Usar Laplacian do OpenCV
                laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
                variance = laplacian.var()
            else:
                # Aproximação simples sem OpenCV
                # Calcular gradiente
                gx = np.diff(img_array, axis=1)
                gy = np.diff(img_array, axis=0)
                variance = np.var(gx) + np.var(gy)
            
            return float(variance)
            
        except Exception as e:
            logger.debug(f"⚠️ Erro ao calcular blur: {e}")
            return 0.0
    
    def _calculate_brightness(self, image: Image.Image) -> float:
        """Calcula brilho médio da imagem (0-1)."""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if NUMPY_AVAILABLE:
                img_array = np.array(image)
                # Calcular luminância
                luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
                return float(np.mean(luminance) / 255.0)
            else:
                # Fallback simples
                grayscale = image.convert('L')
                histogram = grayscale.histogram()
                pixels = sum(histogram)
                brightness = sum(i * w for i, w in enumerate(histogram)) / pixels
                return brightness / 255.0
                
        except Exception as e:
            logger.debug(f"⚠️ Erro ao calcular brilho: {e}")
            return 0.5
    
    def _calculate_contrast(self, image: Image.Image) -> float:
        """Calcula contraste da imagem (0-1)."""
        try:
            if NUMPY_AVAILABLE:
                gray = image.convert('L')
                img_array = np.array(gray)
                return float(np.std(img_array) / 128.0)  # Normalizado
            else:
                # Fallback
                grayscale = image.convert('L')
                histogram = grayscale.histogram()
                pixels = sum(histogram)
                mean = sum(i * w for i, w in enumerate(histogram)) / pixels
                variance = sum(w * (i - mean) ** 2 for i, w in enumerate(histogram)) / pixels
                return min(1.0, (variance ** 0.5) / 128.0)
                
        except Exception as e:
            logger.debug(f"⚠️ Erro ao calcular contraste: {e}")
            return 0.5
    
    def _calculate_sharpness(self, image: Image.Image) -> float:
        """Calcula nitidez da imagem (0-1)."""
        try:
            # Usar filtro de detecção de bordas
            edges = image.convert('L').filter(ImageFilter.FIND_EDGES)
            
            if NUMPY_AVAILABLE:
                edge_array = np.array(edges)
                return float(np.mean(edge_array) / 255.0)
            else:
                histogram = edges.histogram()
                pixels = sum(histogram)
                mean = sum(i * w for i, w in enumerate(histogram)) / pixels
                return mean / 255.0
                
        except Exception as e:
            logger.debug(f"⚠️ Erro ao calcular nitidez: {e}")
            return 0.5
    
    def _detect_faces(self, image: Image.Image) -> int:
        """Detecta faces na imagem usando OpenCV."""
        if not CV2_AVAILABLE:
            return 0
        
        try:
            # Converter para OpenCV
            img_array = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Carregar classificador
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detectar faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return len(faces)
            
        except Exception as e:
            logger.debug(f"⚠️ Erro ao detectar faces: {e}")
            return 0
    
    def _analyze_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> list:
        """Extrai cores dominantes da imagem."""
        if not NUMPY_AVAILABLE:
            return []
        
        try:
            # Redimensionar para acelerar
            img = image.copy()
            img.thumbnail((150, 150))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # K-means simples
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_.astype(int)
                
                return [
                    {"rgb": tuple(c.tolist()), "hex": '#{:02x}{:02x}{:02x}'.format(*c)}
                    for c in colors
                ]
            except ImportError:
                # Fallback: cores mais frequentes
                unique, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
                top_indices = np.argsort(counts)[-n_colors:]
                return [
                    {"rgb": tuple(unique[i].tolist()), "hex": '#{:02x}{:02x}{:02x}'.format(*unique[i])}
                    for i in top_indices
                ]
                
        except Exception as e:
            logger.debug(f"⚠️ Erro ao analisar cores: {e}")
            return []
    
    def _extract_exif(self, image: Image.Image) -> dict:
        """Extrai metadados EXIF da imagem, focado em localização e câmera."""
        exif_data = {}
        
        try:
            exif = image._getexif()
            if not exif:
                return {}
            
            # Tags interessantes
            interesting_tags = [
                'Make', 'Model', 'DateTime', 'DateTimeOriginal',
                'ExposureTime', 'FNumber', 'ISOSpeedRatings',
                'FocalLength', 'Software'
            ]
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if tag_name in interesting_tags:
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    exif_data[tag_name] = value

            # GPS Avançado com piexif
            if PIEXIF_AVAILABLE and 'exif' in image.info:
                try:
                    exif_dict = piexif.load(image.info['exif'])
                    if "GPS" in exif_dict and exif_dict["GPS"]:
                        gps_data = exif_dict["GPS"]
                        lat = self._get_lat_lon(gps_data, piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLatitudeRef)
                        lon = self._get_lat_lon(gps_data, piexif.GPSIFD.GPSLongitude, piexif.GPSIFD.GPSLongitudeRef)
                        if lat is not None and lon is not None:
                            exif_data['GPS_Coordinates'] = f"{lat:.6f}, {lon:.6f}"
                            exif_data['Google_Maps'] = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
                except Exception as e:
                    logger.debug(f"⚠️ Erro no GPS (piexif): {e}")

            return exif_data
            
        except Exception as e:
            logger.debug(f"⚠️ Erro ao extrair EXIF genérico: {e}")
            return {}

    def _get_lat_lon(self, gps_info, key, ref_key):
        """Converte dados GPS EXIF (graus, minutos, seg) para decimal."""
        try:
            if key not in gps_info or ref_key not in gps_info:
                return None
            coords = gps_info[key]
            ref = gps_info[ref_key].decode('utf-8')
            
            d0 = coords[0][0]
            d1 = coords[0][1]
            degrees = float(d0) / float(d1)
            
            m0 = coords[1][0]
            m1 = coords[1][1]
            minutes = float(m0) / float(m1)
            
            s0 = coords[2][0]
            s1 = coords[2][1]
            seconds = float(s0) / float(s1)
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal
        except:
            return None
    
    def _auto_enhance(
        self, 
        image: Image.Image, 
        quality_report: ImageQualityReport
    ) -> Tuple[Image.Image, bool]:
        """Aplica melhorias automáticas baseadas no relatório de qualidade."""
        was_enhanced = False
        
        try:
            # Correção de brilho
            if quality_report.brightness < self.BRIGHTNESS_LOW:
                factor = self.BRIGHTNESS_LOW / max(quality_report.brightness, 0.1)
                factor = min(factor, 2.0)  # Limitar
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)
                was_enhanced = True
                logger.debug(f"💡 Brilho aumentado (fator: {factor:.2f})")
            
            elif quality_report.brightness > self.BRIGHTNESS_HIGH:
                factor = self.BRIGHTNESS_HIGH / quality_report.brightness
                factor = max(factor, 0.5)  # Limitar
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(factor)
                was_enhanced = True
                logger.debug(f"💡 Brilho reduzido (fator: {factor:.2f})")
            
            # Correção de contraste
            if quality_report.contrast < 0.3:
                factor = 1.5
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(factor)
                was_enhanced = True
                logger.debug(f"🎚️ Contraste aumentado (fator: {factor})")
            
            # Sharpening leve se borrado
            if quality_report.is_blurry and quality_report.blur_score > 0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.5)
                was_enhanced = True
                logger.debug("🔍 Nitidez aumentada")
            
            return image, was_enhanced
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao aplicar melhorias: {e}")
            return image, False

    def _deskew_image(self, image: Image.Image) -> Image.Image:
        """Corrige inclinação (skew) da imagem usando Hough lines."""
        try:
            import numpy as np
            
            img_array = np.array(image.convert('L'))
            
            # Detectar bordas
            edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
            
            # Detectar linhas com Hough
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=100, minLineLength=100, maxLineGap=10
            )
            
            if lines is None or len(lines) == 0:
                return image
            
            # Calcular ângulos das linhas
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Filtrar apenas ângulos próximos de 0° (linhas quase horizontais)
                if abs(angle) < 15:
                    angles.append(angle)
            
            if not angles:
                return image
            
            # Ângulo mediano
            median_angle = np.median(angles)
            
            # Se a inclinação for insignificante, não rotacionar
            if abs(median_angle) < 0.5:
                return image
            
            # Rotacionar
            img_rgb = np.array(image.convert('RGB'))
            h, w = img_rgb.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                img_rgb, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            logger.debug(f"📐 Deskew aplicado: {median_angle:.2f}°")
            return Image.fromarray(rotated)
            
        except Exception as e:
            logger.debug(f"⚠️ Erro no deskew: {e}")
            return image

    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Redução de ruído usando Non-Local Means Denoising."""
        try:
            import numpy as np
            
            img_array = np.array(image.convert('RGB'))
            
            # fastNlMeansDenoisingColored para imagens coloridas
            denoised = cv2.fastNlMeansDenoisingColored(
                img_array, None,
                h=6,           # filtro de luminância (menor = menos agressivo)
                hForColorComponents=6,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            logger.debug("🔇 Redução de ruído aplicada")
            return Image.fromarray(denoised)
            
        except Exception as e:
            logger.debug(f"⚠️ Erro na redução de ruído: {e}")
            return image

    def _upscale_image(self, image: Image.Image) -> Image.Image:
        """Upscale de imagens de baixa resolução."""
        w, h = image.size
        
        # Só fazer upscale se imagem for muito pequena
        if min(w, h) >= self.UPSCALE_MIN_PIXELS:
            return image
        
        # Calcular fator de escala (2x ou 3x)
        scale = 2
        if min(w, h) < 320:
            scale = 3
        
        new_w = w * scale
        new_h = h * scale
        
        # Usar LANCZOS para melhor qualidade
        upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        logger.debug(f"🔍 Upscale {scale}x: {w}x{h} → {new_w}x{new_h}")
        return upscaled


def quick_analyze(image: Image.Image) -> Dict[str, Any]:
    """Análise rápida de qualidade para uso simples."""
    preprocessor = ImagePreprocessor(
        auto_rotate=False,
        auto_enhance=False,
        detect_blur=True,
        detect_faces=False,
        analyze_colors=False
    )
    
    result = preprocessor.process(image)
    
    if result.quality_report:
        return {
            "is_blurry": result.quality_report.is_blurry,
            "blur_score": result.quality_report.blur_score,
            "brightness": result.quality_report.brightness,
            "contrast": result.quality_report.contrast,
            "sharpness": result.quality_report.sharpness,
            "recommendations": result.quality_report.recommendations,
            "exif_data": result.quality_report.exif_data
        }
    
    return {}


def auto_fix_image(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """Corrige automaticamente problemas comuns da imagem."""
    preprocessor = ImagePreprocessor(
        auto_rotate=True,
        auto_enhance=True,
        detect_blur=True,
        detect_faces=False,
        analyze_colors=False,
        deskew=True,
        denoise=True,
        upscale=True,
    )
    
    result = preprocessor.process(image)
    
    changes = {
        "rotated": result.was_rotated,
        "enhanced": result.was_enhanced,
        "original_size": result.original_size,
        "final_size": result.final_size
    }
    
    if result.quality_report:
        changes["quality"] = {
            "blur_score": result.quality_report.blur_score,
            "brightness": result.quality_report.brightness,
            "contrast": result.quality_report.contrast
        }
    
    return result.image, changes


def binarize_for_ocr(image: Image.Image) -> Image.Image:
    """Binariza imagem para OCR (Otsu ou adaptativo)."""
    if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
        return image.convert('L')
    
    import numpy as np
    
    # Converter para grayscale
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image
    
    img_array = np.array(gray)
    
    # Binarização adaptativa (melhor para documentos com iluminação desigual)
    binary = cv2.adaptiveThreshold(
        img_array, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    return Image.fromarray(binary)
