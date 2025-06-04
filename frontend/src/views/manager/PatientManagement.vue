<template>
    <div class="patient-management">
        <el-card class="patient-list">
            <template #header>
                <div class="card-header">
                    <span>患者列表</span>
                    <el-button type="primary" @click="handleAddPatient">
                        添加患者
                    </el-button>
                </div>
            </template>

            <!-- 搜索栏 -->
            <div class="search-bar">
                <el-input v-model="searchQuery" placeholder="搜索患者姓名/身份证号" clearable @clear="handleSearch"
                    @input="handleSearch">
                    <template #prefix>
                        <el-icon>
                            <Search />
                        </el-icon>
                    </template>
                </el-input>
            </div>

            <!-- 患者列表表格 -->
            <el-table :data="filteredPatientList" style="width: 100%" v-loading="loading">
                <el-table-column prop="id" label="患者ID" width="120" />
                <el-table-column prop="name" label="姓名" width="120" />
                <el-table-column prop="age" label="年龄" width="80" />
                <el-table-column prop="sex" label="性别" width="80">
                    <template #default="scope">
                        <el-tag :type="scope.row.sex === '男' ? 'primary' : 'success'" size="small">
                            {{ scope.row.sex }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="id_number" label="身份证号" width="180" />
                <el-table-column label="操作" fixed="right" width="100">
                    <template #default="scope">
                        <el-button type="primary" link @click="handleView(scope.row)">
                            查看
                        </el-button>
                    </template>
                </el-table-column>
            </el-table>

            <!-- 分页 -->
            <div class="pagination">
                <el-pagination v-model:current-page="currentPage" v-model:page-size="pageSize"
                    :page-sizes="[10, 20, 30, 50]" layout="total, sizes, prev, pager, next, jumper" :total="total"
                    @size-change="handleSizeChange" @current-change="handleCurrentChange" />
            </div>
        </el-card>

        <!-- 添加患者对话框 -->
        <el-dialog v-model="dialogVisible" title="添加患者" width="50%" :before-close="handleClose">
            <el-form ref="patientFormRef" :model="patientForm" :rules="patientRules" label-width="100px">
                <el-form-item label="姓名" prop="name">
                    <el-input v-model="patientForm.name" placeholder="请输入患者姓名" />
                </el-form-item>

                <el-form-item label="性别" prop="sex">
                    <el-radio-group v-model="patientForm.sex">
                        <el-radio label="男">男</el-radio>
                        <el-radio label="女">女</el-radio>
                    </el-radio-group>
                </el-form-item>

                <el-form-item label="年龄" prop="age">
                    <el-input-number v-model="patientForm.age" :min="0" :max="150" placeholder="请输入年龄" />
                </el-form-item>

                <el-form-item label="身份证号" prop="id_number">
                    <el-input v-model="patientForm.id_number" placeholder="请输入身份证号" />
                </el-form-item>

                <el-form-item label="照片" prop="photo">
                    <el-upload class="avatar-uploader" action="#" :auto-upload="false" :show-file-list="false"
                        :on-change="handlePhotoChange">
                        <img v-if="imageUrl" :src="imageUrl" class="avatar" />
                        <el-icon v-else class="avatar-uploader-icon">
                            <Plus />
                        </el-icon>
                    </el-upload>
                </el-form-item>
            </el-form>

            <template #footer>
                <span class="dialog-footer">
                    <el-button @click="dialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="submitPatient">确定</el-button>
                </span>
            </template>
        </el-dialog>

        <!-- 查看患者详情对话框 -->
        <el-dialog v-model="detailsVisible" title="患者详情" width="600px">
            <div v-if="selectedPatient" class="patient-details">
                <div class="detail-item">
                    <span class="label">姓名：</span>
                    <span class="value">{{ selectedPatient.name }}</span>
                </div>
                <div class="detail-item">
                    <span class="label">性别：</span>
                    <span class="value">
                        <el-tag :type="selectedPatient.sex === '男' ? 'primary' : 'success'" size="small">
                            {{ selectedPatient.sex }}
                        </el-tag>
                    </span>
                </div>
                <div class="detail-item">
                    <span class="label">年龄：</span>
                    <span class="value">{{ selectedPatient.age }}岁</span>
                </div>
                <div class="detail-item">
                    <span class="label">身份证号：</span>
                    <span class="value">{{ selectedPatient.id_number }}</span>
                </div>
                <div class="detail-item">
                    <span class="label">照片：</span>
                    <div class="value">
                        <el-image 
                            v-if="selectedPatient.photo_url" 
                            :src="getImageUrl(selectedPatient.photo_url)"
                            :preview-src-list="[getImageUrl(selectedPatient.photo_url)]"
                            fit="cover"
                            class="patient-photo"
                            @error="handleImageError"
                        >
                            <template #error>
                                <div class="image-error">
                                    <el-icon><Picture /></el-icon>
                                    <span>加载失败</span>
                                </div>
                            </template>
                        </el-image>
                        <span v-else class="no-photo">暂无照片</span>
                    </div>
                </div>
                <div class="detail-item">
                    <span class="label">创建时间：</span>
                    <span class="value">{{ formatDate(selectedPatient.created_at) }}</span>
                </div>
                <div class="detail-item">
                    <span class="label">更新时间：</span>
                    <span class="value">{{ formatDate(selectedPatient.updated_at) }}</span>
                </div>
            </div>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Plus, Picture } from '@element-plus/icons-vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

// 数据
const router = useRouter()
const loading = ref(false)
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const patientList = ref([])
const filteredPatientList = ref([])

// 添加患者相关
const dialogVisible = ref(false)
const detailsVisible = ref(false)
const patientFormRef = ref(null)
const selectedPatient = ref(null)
const imageUrl = ref('')

const patientForm = ref({
    name: '',
    sex: '男',
    age: '',
    id_number: '',
    photo: null
})

const patientRules = {
    name: [
        { required: true, message: '请输入患者姓名', trigger: 'blur' },
        { min: 2, max: 20, message: '长度在 2 到 20 个字符', trigger: 'blur' }
    ],
    sex: [
        { required: true, message: '请选择性别', trigger: 'change' }
    ],
    age: [
        { required: true, message: '请输入年龄', trigger: 'blur' },
        { type: 'number', message: '年龄必须为数字', trigger: 'blur' }
    ],
    id_number: [
        { required: true, message: '请输入身份证号', trigger: 'blur' },
        { pattern: /(^\d{15}$)|(^\d{18}$)|(^\d{17}(\d|X|x)$)/, message: '请输入正确的身份证号', trigger: 'blur' }
    ]
}

// 获取患者列表
const getPatientList = async () => {
    loading.value = true
    try {
        const token = localStorage.getItem('access_token')
        const response = await axios.get('http://localhost:5000/api/patients', {
            headers: {
                'Authorization': `Bearer ${token}`
            },
            params: {
                page: currentPage.value,
                per_page: pageSize.value
            }
        })

        if (response.data.success) {
            patientList.value = response.data.patients
            filteredPatientList.value = patientList.value
            total.value = response.data.pagination.total
        }
    } catch (error) {
        console.error('获取患者列表失败:', error)
        ElMessage.error('获取患者列表失败')
    } finally {
        loading.value = false
    }
}

// 搜索处理
const handleSearch = () => {
    if (!searchQuery.value) {
        filteredPatientList.value = patientList.value
    } else {
        const query = searchQuery.value.toLowerCase()
        filteredPatientList.value = patientList.value.filter(patient =>
            patient.name.toLowerCase().includes(query) ||
            patient.id_number.toLowerCase().includes(query)
        )
    }
    currentPage.value = 1
    total.value = filteredPatientList.value.length
}

// 分页处理
const handleSizeChange = (val) => {
    pageSize.value = val
    getPatientList()
}

const handleCurrentChange = (val) => {
    currentPage.value = val
    getPatientList()
}

// 处理照片变化
const handlePhotoChange = (file) => {
    const isImage = file.raw.type.startsWith('image/')
    const isLt2M = file.raw.size / 1024 / 1024 < 2

    if (!isImage) {
        ElMessage.error('只能上传图片文件!')
        return
    }
    if (!isLt2M) {
        ElMessage.error('图片大小不能超过 2MB!')
        return
    }

    patientForm.value.photo = file.raw
    imageUrl.value = URL.createObjectURL(file.raw)
}

// 添加患者
const handleAddPatient = () => {
    dialogVisible.value = true
    patientForm.value = {
        name: '',
        sex: '男',
        age: '',
        id_number: '',
        photo: null
    }
    imageUrl.value = ''
}

// 查看患者详情
const handleView = (row) => {
    selectedPatient.value = row
    detailsVisible.value = true
}

// 关闭对话框
const handleClose = (done) => {
    ElMessageBox.confirm('确认关闭？')
        .then(() => {
            done()
        })
        .catch(() => {})
}

// 提交患者信息
const submitPatient = async () => {
    if (!patientFormRef.value) return

    await patientFormRef.value.validate(async (valid) => {
        if (valid) {
            try {
                const formData = new FormData()
                formData.append('patient_name', patientForm.value.name)
                formData.append('sex', patientForm.value.sex)
                formData.append('age', String(patientForm.value.age))
                formData.append('id_number', patientForm.value.id_number)
                if (patientForm.value.photo) {
                    formData.append('photo', patientForm.value.photo)
                }

                console.log('提交的数据:', {
                    patient_name: patientForm.value.name,
                    sex: patientForm.value.sex,
                    age: patientForm.value.age,
                    id_number: patientForm.value.id_number,
                    hasPhoto: !!patientForm.value.photo
                })

                const token = localStorage.getItem('access_token')
                const response = await axios.post('http://localhost:5000/api/patients', formData, {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'multipart/form-data'
                    }
                })

                if (response.data.success) {
                    ElMessage.success('添加患者成功')
                    dialogVisible.value = false
                    getPatientList()
                }
            } catch (error) {
                console.error('操作失败:', error)
                console.error('错误响应:', error.response?.data)
                ElMessage.error(error.response?.data?.message || '添加患者失败')
            }
        }
    })
}

// 格式化日期
const formatDate = (dateString) => {
    if (!dateString) return ''
    const date = new Date(dateString)
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    })
}

// 处理图片URL
const getImageUrl = (url) => {
    if (!url) return ''
    if (url.startsWith('http')) return url
    return `http://localhost:5000${url}`
}

// 处理图片加载错误
const handleImageError = () => {
    console.error('图片加载失败:', selectedPatient.value?.photo_url)
}

onMounted(() => {
    getPatientList()
})
</script>

<style scoped>
.patient-management {
    padding: 20px;
}

.patient-list {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 20px;
}

.search-bar {
    padding: 20px;
    border-bottom: 1px solid #ebeef5;
}

.search-bar .el-input {
    width: 300px;
}

.pagination {
    padding: 20px;
    display: flex;
    justify-content: flex-end;
}

.patient-details {
    padding: 20px;
}

.detail-item {
    margin-bottom: 16px;
    display: flex;
    align-items: flex-start;
}

.detail-item .label {
    width: 100px;
    color: #606266;
    font-weight: bold;
}

.detail-item .value {
    color: #303133;
    flex: 1;
}

.patient-photo {
    width: 200px;
    height: 200px;
    border-radius: 4px;
    border: 1px solid #ebeef5;
}

.no-photo {
    color: #909399;
    font-style: italic;
}

.image-error {
    width: 200px;
    height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background-color: #f5f7fa;
    border-radius: 4px;
    color: #909399;
}

.image-error .el-icon {
    font-size: 32px;
    margin-bottom: 8px;
}

.avatar-uploader {
    border: 1px dashed #d9d9d9;
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    width: 178px;
    height: 178px;
}

.avatar-uploader:hover {
    border-color: #409eff;
}

.avatar-uploader-icon {
    font-size: 28px;
    color: #8c939d;
    width: 178px;
    height: 178px;
    text-align: center;
    line-height: 178px;
}

.avatar {
    width: 178px;
    height: 178px;
    display: block;
    object-fit: cover;
}

:deep(.el-upload) {
    width: 100%;
    height: 100%;
}
</style> 