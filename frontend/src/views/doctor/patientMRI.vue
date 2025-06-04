<template>
    <div class="patient-mri">
        <el-card class="mri-list">
            <template #header>
                <div class="card-header">
                    <div class="header-left">
                        <span class="title">MRI序列管理</span>
                        <span class="patient-info" v-if="patientInfo">
                            - 患者：{{ patientInfo.name }}
                        </span>
                    </div>
                    <div class="header-right">
                        <el-button type="primary" @click="handleImport">
                            导入MRI序列
                        </el-button>
                        <el-button @click="$router.back()">返回</el-button>
                    </div>
                </div>
            </template>

            <!-- 序列列表 -->
            <div v-loading="loading">
                <div v-if="sequences.length > 0">
                    <el-table :data="sequences" style="width: 100%">
                        <el-table-column prop="name" label="序列名称" />
                        <el-table-column prop="created_at" label="创建时间">
                            <template #default="scope">
                                {{ new Date(scope.row.created_at).toLocaleString() }}
                            </template>
                        </el-table-column>
                        <el-table-column label="图像数量">
                            <template #default="scope">
                                RGB: {{ scope.row.rgb_count }} | 深度: {{ scope.row.depth_count }}
                            </template>
                        </el-table-column>
                        <el-table-column label="操作" width="200">
                            <template #default="scope">
                                <el-button type="primary" link @click="handleViewSequence(scope.row)">
                                    查看
                                </el-button>
                            </template>
                        </el-table-column>
                    </el-table>
                </div>
                <el-empty v-else description="暂无MRI序列" />
            </div>
        </el-card>

        <!-- 导入对话框 -->
        <el-dialog v-model="importDialogVisible" title="导入MRI序列" width="600px" :close-on-click-modal="false">
            <el-form ref="importFormRef" :model="importForm" :rules="importRules" label-width="100px">
                <el-form-item label="序列名称" prop="seq_name">
                    <el-input v-model="importForm.seq_name" placeholder="请输入序列名称" />
                </el-form-item>
                <el-form-item label="RGB图像" prop="rgb_files">
                    <el-upload ref="rgbUploadRef" action="#" :auto-upload="false" :multiple="true"
                        :show-file-list="true" :on-change="handleRGBChange" accept=".jpg,.jpeg,.png,.bmp">
                        <template #trigger>
                            <el-button type="primary">选择RGB图像</el-button>
                        </template>
                        <template #tip>
                            <div class="el-upload__tip">
                                支持jpg、jpeg、png、bmp格式
                            </div>
                        </template>
                    </el-upload>
                </el-form-item>
                <el-form-item label="深度图像" prop="depth_files">
                    <el-upload ref="depthUploadRef" action="#" :auto-upload="false" :multiple="true"
                        :show-file-list="true" :on-change="handleDepthChange" accept=".jpg,.jpeg,.png,.bmp">
                        <template #trigger>
                            <el-button type="primary">选择深度图像</el-button>
                        </template>
                        <template #tip>
                            <div class="el-upload__tip">
                                支持jpg、jpeg、png、bmp格式
                            </div>
                        </template>
                    </el-upload>
                </el-form-item>
            </el-form>
            <template #footer>
                <span class="dialog-footer">
                    <el-button @click="importDialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="submitImport" :loading="importing">
                        确认导入
                    </el-button>
                </span>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

const route = useRoute()
const router = useRouter()
const loading = ref(false)
const sequences = ref([])
const patientInfo = ref(null)

// 导入相关的响应式变量
const importDialogVisible = ref(false)
const importing = ref(false)
const importFormRef = ref(null)
const rgbUploadRef = ref(null)
const depthUploadRef = ref(null)
const importForm = ref({
    seq_name: '',
    rgb_files: [],
    depth_files: []
})

// 表单验证规则
const importRules = {
    seq_name: [
        { required: true, message: '请输入序列名称', trigger: 'blur' },
        { min: 1, max: 50, message: '长度在1到50个字符', trigger: 'blur' }
    ],
    rgb_files: [
        { required: true, message: '请选择RGB图像', trigger: 'change' }
    ],
    depth_files: [
        { required: true, message: '请选择深度图像', trigger: 'change' }
    ]
}

// 处理RGB图像选择
const handleRGBChange = (file, fileList) => {
    importForm.value.rgb_files = fileList
}

// 处理深度图像选择
const handleDepthChange = (file, fileList) => {
    importForm.value.depth_files = fileList
}

// 打开导入对话框
const handleImport = () => {
    importDialogVisible.value = true
    importForm.value = {
        seq_name: '',
        rgb_files: [],
        depth_files: []
    }
    if (rgbUploadRef.value) {
        rgbUploadRef.value.clearFiles()
    }
    if (depthUploadRef.value) {
        depthUploadRef.value.clearFiles()
    }
}

// 提交导入
const submitImport = async () => {
    if (!importFormRef.value) return

    try {
        await importFormRef.value.validate()

        // 检查文件数量是否匹配
        if (importForm.value.rgb_files.length !== importForm.value.depth_files.length) {
            ElMessage.error('RGB图像和深度图像数量必须相同')
            return
        }

        importing.value = true
        const formData = new FormData()
        formData.append('seq_name', importForm.value.seq_name)

        // 添加RGB图像
        importForm.value.rgb_files.forEach(file => {
            formData.append('rgb_files[]', file.raw)
        })

        // 添加深度图像
        importForm.value.depth_files.forEach(file => {
            formData.append('depth_files[]', file.raw)
        })

        const response = await axios.post(
            `${import.meta.env.VITE_API_URL}/api/mri/patients/${route.params.patientId}/sequences`,
            formData,
            {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
                    'Content-Type': 'multipart/form-data'
                }
            }
        )

        if (response.data.success) {
            ElMessage.success('序列导入成功')
            importDialogVisible.value = false
            fetchSequences() // 刷新序列列表
        } else {
            ElMessage.error(response.data.message || '序列导入失败')
        }
    } catch (error) {
        console.error('导入序列失败:', error)
        ElMessage.error(error.response?.data?.message || '导入失败，请稍后重试')
    } finally {
        importing.value = false
    }
}

// 获取序列列表
const fetchSequences = async () => {
    try {
        loading.value = true
        const patientId = route.params.patientId
        console.log('正在获取MRI序列，患者ID:', patientId)
        console.log('请求URL:', `${import.meta.env.VITE_API_URL}/api/mri/patients/${patientId}/sequences`)

        const response = await axios.get(`${import.meta.env.VITE_API_URL}/api/mri/patients/${patientId}/sequences`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        })
        console.log('API响应:', response.data)

        if (response.data.success) {
            sequences.value = response.data.sequences
            patientInfo.value = response.data.patient
        } else {
            ElMessage.error(response.data.message || '获取序列列表失败')
        }
    } catch (error) {
        console.error('获取序列列表失败:', error)
        console.error('错误详情:', {
            message: error.message,
            config: error.config,
            response: error.response
        })
        ElMessage.error('获取序列列表失败，请稍后重试')
    } finally {
        loading.value = false
    }
}

// 查看序列详情
const handleViewSequence = (sequence) => {
    router.push({
        name: 'sequenceDetail',
        params: {
            patientId: route.params.patientId,
            sequenceId: sequence.id
        }
    })
}

onMounted(() => {
    fetchSequences()
})
</script>

<style scoped>
.patient-mri {
    padding: 20px;
}

.mri-list {
    max-width: 1200px;
    margin: 0 auto;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.title {
    font-size: 18px;
    font-weight: bold;
}

.patient-info {
    color: #666;
}

.header-right {
    display: flex;
    gap: 10px;
}

.dialog-footer {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
}

:deep(.el-upload-list) {
    max-height: 200px;
    overflow-y: auto;
}
</style>