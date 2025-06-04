<template>
    <div class="effect-prediction">
        <!-- 文件资源管理器 -->
        <el-card class="file-explorer">
            <template #header>
                <div class="card-header">
                    <span>MRI图像浏览器</span>
                </div>
            </template>

            <!-- 面包屑导航 -->
            <el-breadcrumb v-if="currentPath.length > 0" class="breadcrumb">
                <el-breadcrumb-item @click="navigateTo(-1)">根目录</el-breadcrumb-item>
                <el-breadcrumb-item v-for="(item, index) in currentPath" :key="index" @click="navigateTo(index)">
                    {{ item.name }}
                </el-breadcrumb-item>
            </el-breadcrumb>

            <!-- 文件列表 -->
            <div class="file-list">
                <!-- 显示患者列表 -->
                <template v-if="currentLevel === 0">
                    <div v-for="patient in patientList" :key="patient.id" class="file-item"
                        @click="openPatient(patient)">
                        <el-icon>
                            <Folder />
                        </el-icon>
                        <span>{{ patient.name }}</span>
                    </div>
                </template>

                <!-- 显示序列列表 -->
                <template v-else-if="currentLevel === 1">
                    <div v-for="sequence in sequences" :key="sequence.id" class="file-item"
                        @click="openSequence(sequence)">
                        <el-icon>
                            <Folder />
                        </el-icon>
                        <span>{{ sequence.name }}</span>
                    </div>
                </template>

                <!-- 显示图像类型 -->
                <template v-else-if="currentLevel === 2">
                    <div class="image-types">
                        <div class="file-item" @click="selectImageType('rgb')">
                            <el-icon>
                                <Picture />
                            </el-icon>
                            <span>RGB图像</span>
                        </div>
                        <div class="file-item" @click="selectImageType('depth')">
                            <el-icon>
                                <Picture />
                            </el-icon>
                            <span>深度图像</span>
                        </div>
                    </div>

                    <!-- 缩略图列表 -->
                    <div v-if="showThumbnails" class="thumbnails-container">
                        <div v-for="index in maxImageIndex" :key="index" class="thumbnail-item"
                            :class="{ active: currentImageIndex === index - 1 }" @click="selectThumbnail(index - 1)">
                            <img :src="thumbnailUrls[index - 1]" :alt="`图像 ${index}`" />
                        </div>
                    </div>
                </template>
            </div>
        </el-card>

        <!-- 裁剪与布针区域 -->
        <el-card class="operation-area">
            <template #header>
                <div class="card-header">
                    <span>{{ operationTitle }}</span>
                    <div class="operation-controls" v-if="showPreview">
                        <el-button-group v-if="!isCropping && !isNeedling && !showPredictButton">
                            <el-button type="primary" @click="startCrop">开始裁剪</el-button>
                        </el-button-group>
                        <el-button-group v-if="isCropping">
                            <el-button type="success" @click="confirmCrop">确认裁剪</el-button>
                            <el-button @click="cancelCrop">取消</el-button>
                        </el-button-group>
                        <el-button-group v-if="isNeedling">
                            <el-button type="success" @click="confirmNeedling">确认布针</el-button>
                            <el-button @click="cancelNeedling">取消</el-button>
                        </el-button-group>
                        <el-button-group v-if="showPredictButton">
                            <el-button type="primary" :loading="predicting" @click="predict">开始预测</el-button>
                            <el-button @click="showPredictButton = false">返回</el-button>
                        </el-button-group>
                    </div>
                </div>
            </template>

            <!-- 图像显示和操作区域 -->
            <div class="operation-content" v-if="showPreview">
                <div class="image-container" v-loading="imageLoading">
                    <div v-if="isCropping" class="cropper-container">
                        <img ref="cropperImage" :src="currentImage" style="max-width: 100%;" alt="待裁剪图像" />
                    </div>
                    <div v-else-if="isNeedling" class="needling-area" ref="needlingArea" @click="handleNeedleClick">
                        <img :src="croppedImage" alt="裁剪后的图像" />
                        <div v-for="(point, index) in needlePoints" :key="index" class="needle-point"
                            :style="{ left: point.x + 'px', top: point.y + 'px' }"></div>
                    </div>
                    <img v-else :src="currentImage" alt="MRI图像" class="preview-image" />
                </div>

                <!-- 操作提示 -->
                <div class="operation-tips" v-if="isCropping">
                    请裁剪352*352像素的区域
                </div>
                <div class="operation-tips" v-if="isNeedling">
                    请在图像上点击添加布针点，右键可删除布针点
                </div>
            </div>
        </el-card>

        <!-- 预测结果显示区域 -->
        <el-card class="result-area">
            <template #header>
                <div class="card-header">
                    <span>预测结果</span>
                </div>
            </template>

            <div class="result-content" v-if="predictionResult">
                <div class="result-images">
                    <div class="result-image-item">
                        <h4>布针图</h4>
                        <div class="result-image-container">
                            <img :src="predictionResult.needledImageUrl" alt="布针图" class="result-image"
                                v-loading="predicting" />
                        </div>
                    </div>
                    <div class="result-image-item">
                        <h4>预测结果</h4>
                        <div class="result-image-container">
                            <img :src="predictionResult.resultUrl" alt="预测结果" class="result-image"
                                v-loading="predicting" />
                        </div>
                    </div>
                </div>
                <div class="result-info">
                    <p>预测时间：{{ predictionResult.predTime }}</p>
                </div>
            </div>
            <div v-else class="no-result">
                <el-empty description="暂无预测结果"></el-empty>
            </div>
        </el-card>
    </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { Folder, Picture } from '@element-plus/icons-vue'
import axios from 'axios'
import 'cropperjs/dist/cropper.min.css'
import Cropper from 'cropperjs/dist/cropper.min.js'

const currentLevel = ref(0) // 0: 患者列表, 1: 序列列表, 2: 图像类型
const currentPath = ref([]) // 当前路径
const patientList = ref([])
const sequences = ref([])
const currentImageIndex = ref(0)
const selectedSequence = ref(null)
const currentImage = ref('')
const imageLoading = ref(false)
const currentImageType = ref('rgb')
const showPreview = ref(false)
const showThumbnails = ref(false)
const imageItemIds = ref([])  // 存储图像的item_id

const predictionForm = ref({
    patientId: '',
    sequenceId: '',
    psa: null,
})

const rules = {
    psa: [
        { required: true, message: '请输入PSA值', trigger: 'blur' }
    ],
}

const isCropping = ref(false)
const isNeedling = ref(false)
const cropperImage = ref(null)
let cropper = null
const needlingArea = ref(null)
const croppedImage = ref('')
const needlePoints = ref([])
const cropPosition = ref(null)
const showPredictButton = ref(false)
const predicting = ref(false)
const thumbnailUrls = ref([])
const predictionResult = ref(null)

// 计算最大图像索引
const maxImageIndex = computed(() => {
    if (!selectedSequence.value) return 0
    return currentImageType.value === 'rgb'
        ? selectedSequence.value.rgb_count
        : selectedSequence.value.depth_count
})

// 导航到指定层级
const navigateTo = (index) => {
    if (index === -1) {
        currentLevel.value = 0
        currentPath.value = []
        sequences.value = []
        showPreview.value = false
    } else {
        currentLevel.value = index + 1
        currentPath.value = currentPath.value.slice(0, index + 1)
        if (index === 0) {
            getSequences(currentPath.value[0].id)
        }
    }
}

// 打开患者文件夹
const openPatient = async (patient) => {
    currentLevel.value = 1
    currentPath.value = [{ id: patient.id, name: patient.name }]
    predictionForm.value.patientId = patient.id
    await getSequences(patient.id)
}

// 打开序列文件夹
const openSequence = async (sequence) => {
    currentLevel.value = 2
    currentPath.value.push({ id: sequence.id, name: sequence.name })
    selectedSequence.value = sequence
    predictionForm.value.sequenceId = sequence.id

    try {
        // 获取序列详情，包括所有图像
        const token = localStorage.getItem('access_token')
        const response = await axios.get(
            `http://localhost:5000/api/mri/patients/${predictionForm.value.patientId}/sequences/${sequence.id}`,
            {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            }
        )

        if (response.data.success) {
            const items = response.data.sequence.items
            // 分别保存RGB和深度图像的item_id
            const rgbIds = items.map(item => item.rgb.id)
            const depthIds = items.map(item => item.depth.id)

            // 保存到localStorage，使用不同的key区分RGB和深度图像
            localStorage.setItem('rgbImageItemIds', JSON.stringify(rgbIds))
            localStorage.setItem('depthImageItemIds', JSON.stringify(depthIds))

            console.log('RGB图像item_id列表:', rgbIds)
            console.log('深度图像item_id列表:', depthIds)
        }
    } catch (error) {
        console.error('获取序列详情失败:', error)
        ElMessage.error('获取序列详情失败')
    }
}

// 选择图像类型
const selectImageType = async (type) => {
    currentImageType.value = type
    currentImageIndex.value = 0
    showPreview.value = true
    showThumbnails.value = true

    try {
        // 获取图像列表
        const token = localStorage.getItem('access_token')
        const response = await axios.get(
            `http://localhost:5000/api/mri/patients/${predictionForm.value.patientId}/sequences/${predictionForm.value.sequenceId}/images`,
            {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            }
        )

        if (response.data.success) {
            // 保存图像的item_id
            imageItemIds.value = response.data.images.map(img => img.id)
            console.log('图像item_id列表:', imageItemIds.value)
            localStorage.setItem('imageItemIds', JSON.stringify(imageItemIds.value))
        }
    } catch (error) {
        console.error('获取图像列表失败:', error)
    }

    // 预加载所有缩略图
    thumbnailUrls.value = []
    for (let i = 0; i < maxImageIndex.value; i++) {
        try {
            const token = localStorage.getItem('access_token')
            const typeParam = type === 'depth' ? '?type=depth' : ''
            const response = await axios.get(
                `http://localhost:5000/api/mri/patients/${predictionForm.value.patientId}/sequences/${predictionForm.value.sequenceId}/images/${i}${typeParam}`,
                {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    },
                    responseType: 'blob'
                }
            )
            const blob = new Blob([response.data], { type: 'image/jpeg' })
            const imageUrl = URL.createObjectURL(blob)
            thumbnailUrls.value.push(imageUrl)
        } catch (error) {
            console.error('获取缩略图失败:', error)
            thumbnailUrls.value.push('')
        }
    }

    fetchCurrentImage()
}

// 获取患者列表
const getPatientList = async () => {
    try {
        const token = localStorage.getItem('access_token')
        const response = await axios.get('http://localhost:5000/api/patients', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        if (response.data.success) {
            patientList.value = response.data.patients
        }
    } catch (error) {
        ElMessage.error('获取患者列表失败')
        console.error(error)
    }
}

// 获取序列列表
const getSequences = async (patientId) => {
    try {
        const token = localStorage.getItem('access_token')
        const response = await axios.get(`http://localhost:5000/api/mri/patients/${patientId}/sequences`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        if (response.data.success) {
            sequences.value = response.data.sequences
        }
    } catch (error) {
        ElMessage.error('获取MRI序列失败')
        console.error(error)
    }
}

// 获取当前图片
const fetchCurrentImage = async () => {
    if (!predictionForm.value.patientId || !predictionForm.value.sequenceId) return

    imageLoading.value = true
    try {
        const token = localStorage.getItem('access_token')
        const typeParam = currentImageType.value === 'depth' ? '?type=depth' : ''
        const response = await axios.get(
            `http://localhost:5000/api/mri/patients/${predictionForm.value.patientId}/sequences/${predictionForm.value.sequenceId}/images/${currentImageIndex.value}${typeParam}`,
            {
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                responseType: 'blob'
            }
        )

        const blob = new Blob([response.data], { type: 'image/jpeg' })
        const imageUrl = URL.createObjectURL(blob)
        currentImage.value = imageUrl

        // 预加载图像
        const img = new Image()
        img.src = imageUrl
        img.onload = () => {
            imageLoading.value = false
        }
    } catch (error) {
        ElMessage.error('获取图片失败')
        console.error(error)
        currentImage.value = ''
    }
}

// 切换到上一张图片
const prevImage = async () => {
    if (currentImageIndex.value > 0) {
        currentImageIndex.value--
        await fetchCurrentImage()
    }
}

// 切换到下一张图片
const nextImage = async () => {
    if (currentImageIndex.value < maxImageIndex.value - 1) {
        currentImageIndex.value++
        await fetchCurrentImage()
    }
}

// 提交预测
const submitForm = async () => {
    if (!predictionFormRef.value) return

    await predictionFormRef.value.validate(async (valid) => {
        if (valid) {
            try {
                // TODO: 调用后端预测API
                ElMessage.success('预测完成')
            } catch (error) {
                ElMessage.error('预测失败')
            }
        }
    })
}

// 重置表单
const resetForm = () => {
    if (predictionFormRef.value) {
        predictionFormRef.value.resetFields()
    }
}

// 计算操作区域标题
const operationTitle = computed(() => {
    if (isCropping.value) return '图像裁剪'
    if (isNeedling.value) return '布针操作'
    return '图像预览'
})

// 开始裁剪
const startCrop = () => {
    if (!currentImage.value) {
        ElMessage.error('请先选择图像')
        return
    }
    console.log('开始裁剪，当前图像URL:', currentImage.value)
    isCropping.value = true

    // 在下一个 tick 初始化 cropper
    nextTick(() => {
        if (cropperImage.value) {
            if (cropper) {
                cropper.destroy()
            }
            cropper = new Cropper(cropperImage.value, {
                viewMode: 1,
                dragMode: 'none',  // 禁止拖动图片，只允许移动裁剪框
                initialAspectRatio: 1,
                aspectRatio: 1,
                cropBoxResizable: false,  // 禁止调整裁剪框大小
                cropBoxMovable: true,  // 允许移动裁剪框
                data: {  // 设置初始裁剪框大小和位置
                    width: 352,
                    height: 352,
                },
                ready: function () {
                    // 在裁剪器准备好后，确保裁剪框大小正确
                    const cropBoxData = this.cropper.getCropBoxData();
                    this.cropper.setCropBoxData({
                        width: 352,
                        height: 352,
                        left: cropBoxData.left,
                        top: cropBoxData.top
                    });
                },
                zoom: function (e) {
                    // 在缩放时保持裁剪框大小不变
                    const cropBoxData = this.cropper.getCropBoxData();
                    this.cropper.setCropBoxData({
                        width: 352,
                        height: 352,
                        left: cropBoxData.left,
                        top: cropBoxData.top
                    });
                }
            })
        }
    })
}

// 确认裁剪
const confirmCrop = () => {
    if (!cropper) {
        console.error('裁剪组件未初始化')
        return
    }

    try {
        console.log('开始获取裁剪数据')
        const canvas = cropper.getCroppedCanvas({
            width: 352,
            height: 352
        })

        // 获取裁剪框的位置数据
        const cropData = cropper.getData()
        cropPosition.value = {
            x: Math.round(cropData.x),
            y: Math.round(cropData.y)
        }
        console.log('裁剪区域信息:', {
            cropPosition: cropPosition.value,
            cropData: {
                x: cropData.x,
                y: cropData.y,
                width: cropData.width,
                height: cropData.height
            }
        })

        croppedImage.value = canvas.toDataURL('image/png')
        console.log('获取到裁剪数据')
        isCropping.value = false
        isNeedling.value = true
        needlePoints.value = []

        // 销毁 cropper 实例
        cropper.destroy()
        cropper = null
    } catch (error) {
        console.error('裁剪失败:', error)
        ElMessage.error('裁剪失败')
    }
}

// 取消裁剪
const cancelCrop = () => {
    if (cropper) {
        cropper.destroy()
        cropper = null
    }
    isCropping.value = false
}

// 处理布针点击
const handleNeedleClick = (event) => {
    if (!needlingArea.value) return

    const img = needlingArea.value.querySelector('img')
    if (!img) return

    // 获取图片的实际显示区域
    const imgRect = img.getBoundingClientRect()
    console.log('图片区域信息:', {
        left: imgRect.left,
        top: imgRect.top,
        width: imgRect.width,
        height: imgRect.height
    })

    // 计算点击位置相对于图片的坐标
    const x = event.clientX - imgRect.left
    const y = event.clientY - imgRect.top
    console.log('原始点击位置:', {
        clientX: event.clientX,
        clientY: event.clientY,
        calculatedX: x,
        calculatedY: y
    })

    // 由于图片固定为352x352，不需要缩放转换
    const cropX = Math.round(x)
    const cropY = Math.round(y)
    console.log('裁剪区域内位置:', {
        cropX,
        cropY
    })

    // 检查是否点击了已有的针点
    const clickedPointIndex = needlePoints.value.findIndex(point => {
        const distance = Math.sqrt(Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2))
        return distance < 5 // 5px的判定范围
    })

    if (event.button === 2) { // 右键删除针点
        if (clickedPointIndex !== -1) {
            needlePoints.value.splice(clickedPointIndex, 1)
        }
    } else { // 左键添加针点
        if (clickedPointIndex === -1 && x >= 0 && x <= 352 && y >= 0 && y <= 352) {
            needlePoints.value.push({
                x: cropX,
                y: cropY,
                cropX,
                cropY
            })
            console.log('添加布针点:', {
                displayPoint: { x: cropX, y: cropY },
                cropPoint: { x: cropX, y: cropY }
            })
        }
    }
}

// 确认布针
const confirmNeedling = async () => {
    if (!croppedImage.value || needlePoints.value.length === 0) {
        ElMessage.error('请先完成布针')
        return
    }

    try {
        // 创建一个canvas来绘制带针点的图片
        const img = new Image()
        img.src = croppedImage.value

        await new Promise((resolve, reject) => {
            img.onload = resolve
            img.onerror = reject
        })

        const canvas = document.createElement('canvas')
        canvas.width = 352
        canvas.height = 352
        const ctx = canvas.getContext('2d')

        // 绘制裁剪后的图片
        ctx.drawImage(img, 0, 0, 352, 352)

        // 绘制针点（使用裁剪图中的实际像素位置）
        needlePoints.value.forEach(point => {
            ctx.beginPath()
            ctx.arc(point.cropX, point.cropY, 3, 0, Math.PI * 2)
            ctx.fillStyle = 'red'
            ctx.fill()
        })

        // 将canvas转换为blob
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'))

        // 创建要保存的数据（使用实际像素位置）
        const saveData = {
            cropPosition: cropPosition.value,
            needlePoints: needlePoints.value.map(point => ({
                x: point.cropX,
                y: point.cropY
            }))
        }
        console.log('保存的布针数据:', saveData)

        // 创建FormData
        const formData = new FormData()
        formData.append('image', blob, 'needled.png')
        formData.append('patient_id', predictionForm.value.patientId)
        formData.append('sequence_id', predictionForm.value.sequenceId)
        formData.append('image_type', currentImageType.value)
        formData.append('needle_points', JSON.stringify(saveData))

        // 发送到服务器
        const token = localStorage.getItem('access_token')
        const response = await axios.post(
            'http://localhost:5000/api/mri/needle',
            formData,
            {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'multipart/form-data'
                }
            }
        )

        if (response.data.success) {
            ElMessage.success('布针图片保存成功')
            isNeedling.value = false
            showPredictButton.value = true  // 显示预测按钮
        } else {
            throw new Error(response.data.message || '保存失败')
        }
    } catch (error) {
        console.error('保存布针图片失败:', error)
        console.error('错误详情:', error.response?.data || error.message)
        ElMessage.error('保存布针图片失败')
    }
}

// 取消布针
const cancelNeedling = () => {
    isNeedling.value = false
    needlePoints.value = []
}

// 添加预测函数
const predict = async () => {
    if (predicting.value) return

    try {
        predicting.value = true
        const token = localStorage.getItem('access_token')

        // 始终使用RGB图像的item_id
        const rgbImageIds = JSON.parse(localStorage.getItem('rgbImageItemIds') || '[]')
        const currentItemId = rgbImageIds[currentImageIndex.value]

        // 确保所有必要的数据都存在
        if (!currentItemId) {
            throw new Error('无法获取当前图像的ID')
        }
        if (!cropPosition.value) {
            throw new Error('请先裁剪图像')
        }
        if (!needlePoints.value || needlePoints.value.length === 0) {
            throw new Error('请先添加布针点')
        }

        // 准备预测数据
        const predictData = {
            item_id: currentItemId,
            crop_position: cropPosition.value,
            needle_positions: needlePoints.value.map(point => ({
                x: point.cropX + cropPosition.value.x,
                y: point.cropY + cropPosition.value.y
            }))
        }

        console.log('预测数据详情:', {
            cropPosition: cropPosition.value,
            originalPoints: needlePoints.value.map(p => ({ x: p.cropX, y: p.cropY })),
            transformedPoints: predictData.needle_positions,
            imageId: currentItemId
        })

        // 发送预测请求
        const response = await axios.post(
            'http://localhost:5000/api/predictions/predict',
            predictData,
            {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            }
        )

        if (response.data.success) {
            ElMessage.success('预测完成')
            console.log('完整的预测结果数据:', response.data)
            console.log('预测结果路径:', response.data.prediction.result_path)

            // 更新预测结果
            const resultPath = response.data.prediction.result_path.replace(/\\/g, '/')
            const resultUrl = `http://localhost:5000/${resultPath}`

            // 重新生成带针点的图像
            const img = new Image()
            img.src = croppedImage.value
            await new Promise((resolve, reject) => {
                img.onload = resolve
                img.onerror = reject
            })

            const canvas = document.createElement('canvas')
            canvas.width = 352
            canvas.height = 352
            const ctx = canvas.getContext('2d')

            // 绘制裁剪后的图片
            ctx.drawImage(img, 0, 0, 352, 352)

            // 绘制针点
            needlePoints.value.forEach(point => {
                ctx.beginPath()
                ctx.arc(point.cropX, point.cropY, 3, 0, Math.PI * 2)
                ctx.fillStyle = 'red'
                ctx.fill()
            })

            // 获取带针点的图像URL
            const needledImageUrl = canvas.toDataURL('image/png')

            console.log('构造的预测结果URL:', resultUrl)
            console.log('生成的布针图URL:', needledImageUrl)

            predictionResult.value = {
                resultUrl: resultUrl,
                needledImageUrl: needledImageUrl,
                predTime: new Date().toLocaleString()
            }
        } else {
            throw new Error(response.data.message || '预测失败')
        }
    } catch (error) {
        console.error('预测失败:', error)
        ElMessage.error('预测失败: ' + (error.response?.data?.message || error.message))
    } finally {
        predicting.value = false
    }
}

// 获取缩略图URL
const getThumbnailUrl = async (index) => {
    if (!predictionForm.value.patientId || !predictionForm.value.sequenceId) return ''

    const token = localStorage.getItem('access_token')
    const typeParam = currentImageType.value === 'depth' ? '?type=depth' : ''
    return `http://localhost:5000/api/mri/patients/${predictionForm.value.patientId}/sequences/${predictionForm.value.sequenceId}/images/${index}${typeParam}`
}

// 选择缩略图
const selectThumbnail = async (index) => {
    currentImageIndex.value = index
    await fetchCurrentImage()
}

// 组件卸载时清理资源
onUnmounted(() => {
    if (cropper) {
        cropper.destroy()
        cropper = null
    }
    if (currentImage.value) {
        URL.revokeObjectURL(currentImage.value)
    }
    if (croppedImage.value && croppedImage.value.startsWith('blob:')) {
        URL.revokeObjectURL(croppedImage.value)
    }
    // 清理缩略图URL
    thumbnailUrls.value.forEach(url => {
        if (url) {
            URL.revokeObjectURL(url)
        }
    })
})

onMounted(() => {
    getPatientList()
    if (needlingArea.value) {
        needlingArea.value.addEventListener('contextmenu', (e) => e.preventDefault())
    }
})
</script>

<style scoped>
.effect-prediction {
    padding: 20px;
    display: flex;
    gap: 24px;
    height: calc(100vh - 100px);
    background-color: #f5f7fa;
}

.file-explorer,
.operation-area,
.result-area {
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.file-explorer:hover,
.operation-area:hover,
.result-area:hover {
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}

.file-explorer {
    width: 25%;
    min-width: 250px;
}

.operation-area {
    width: 33%;
    display: flex;
    flex-direction: column;
}

.card-header {
    padding: 16px 20px;
    border-bottom: 1px solid #ebeef5;
    background-color: #fafafa;
    border-radius: 12px 12px 0 0;
}

.card-header span {
    font-size: 16px;
    font-weight: 600;
    color: #303133;
}

.operation-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 20px;
}

.image-container {
    width: 100%;
    height: 0;
    padding-bottom: 100%;
    position: relative;
    border: 2px solid #ebeef5;
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.3s ease;
}

.image-container:hover {
    border-color: #409eff;
}

.preview-image {
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.needling-area {
    position: absolute;
    width: 352px;
    height: 352px;
    cursor: crosshair;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
    border-radius: 4px;
}

.needling-area img {
    width: 352px;
    height: 352px;
    object-fit: none;
}

.needle-point {
    position: absolute;
    width: 8px;
    height: 8px;
    background-color: #f56c6c;
    border: 2px solid rgba(255, 255, 255, 0.8);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    box-shadow: 0 0 4px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
}

.operation-tips {
    text-align: center;
    color: #606266;
    padding: 12px;
    background-color: #f5f7fa;
    border-radius: 8px;
    font-size: 14px;
    border: 1px solid #ebeef5;
}

.operation-controls {
    display: flex;
    gap: 12px;
}

.file-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    cursor: pointer;
    border-radius: 8px;
    transition: all 0.3s ease;
    color: #606266;
}

.file-item:hover {
    background-color: #f5f7fa;
    color: #409eff;
}

.file-item .el-icon {
    font-size: 18px;
}

.thumbnails-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 12px;
    margin-top: 20px;
    padding: 16px;
    border: 1px solid #ebeef5;
    border-radius: 8px;
    background-color: #fafafa;
}

.thumbnail-item {
    aspect-ratio: 1;
    border: 2px solid transparent;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.3s ease;
    background-color: white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.thumbnail-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(64, 158, 255, 0.2);
}

.thumbnail-item.active {
    border-color: #409eff;
    box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
}

.thumbnail-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.result-content {
    padding: 20px;
}

.result-images {
    display: flex;
    gap: 24px;
    margin-bottom: 24px;
}

.result-image-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: #fafafa;
    padding: 16px;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.result-image-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.result-image-item h4 {
    margin: 0 0 12px 0;
    color: #303133;
    font-size: 15px;
    font-weight: 600;
}

.result-image-container {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    padding: 8px;
}

.result-image {
    max-width: 100%;
    height: auto;
    object-fit: contain;
    border-radius: 4px;
}

.result-info {
    text-align: center;
    color: #606266;
    font-size: 14px;
    background-color: #f5f7fa;
    padding: 12px;
    border-radius: 8px;
    margin-top: 16px;
}

.no-result {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 300px;
    background-color: #fafafa;
    border-radius: 8px;
}

:deep(.el-empty) {
    padding: 40px;
}

:deep(.el-button) {
    border-radius: 6px;
    font-weight: 500;
}

:deep(.el-button--primary) {
    background-color: #409eff;
    border-color: #409eff;

    &:hover {
        background-color: #66b1ff;
        border-color: #66b1ff;
    }
}

:deep(.el-button--success) {
    background-color: #67c23a;
    border-color: #67c23a;

    &:hover {
        background-color: #85ce61;
        border-color: #85ce61;
    }
}

:deep(.el-breadcrumb__item) {
    cursor: pointer;

    .el-breadcrumb__inner {
        color: #606266;

        &:hover {
            color: #409eff;
        }
    }
}
</style>