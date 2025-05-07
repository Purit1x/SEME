<template>
    <div class="doctor-profile">
        <el-card>
            <template #header>
                <div class="card-header">
                    <span>个人信息</span>
                    <el-button type="primary" @click="handleEdit">
                        编辑信息
                    </el-button>
                </div>
            </template>

            <el-form ref="profileFormRef" :model="profileForm" :rules="rules" label-width="100px"
                :disabled="!isEditing">
                <el-form-item label="医生ID">
                    <el-input v-model="profileForm.id" disabled />
                </el-form-item>

                <el-form-item label="姓名" prop="name">
                    <el-input v-model="profileForm.name" />
                </el-form-item>

                <el-form-item label="科室" prop="department">
                    <el-input v-model="profileForm.department" disabled />
                </el-form-item>

                <el-form-item label="邮箱" prop="email">
                    <el-input v-model="profileForm.email" disabled />
                </el-form-item>

                <el-form-item v-if="isEditing">
                    <el-button type="primary" @click="submitForm">
                        保存
                    </el-button>
                    <el-button @click="cancelEdit">取消</el-button>
                </el-form-item>
            </el-form>
        </el-card>

        <!-- 修改密码卡片 -->
        <el-card class="password-card">
            <template #header>
                <div class="card-header">
                    <span>修改密码</span>
                </div>
            </template>

            <el-form ref="passwordFormRef" :model="passwordForm" :rules="passwordRules" label-width="100px">
                <el-form-item label="原密码" prop="oldPassword">
                    <el-input v-model="passwordForm.oldPassword" type="password" show-password />
                </el-form-item>

                <el-form-item label="新密码" prop="newPassword">
                    <el-input v-model="passwordForm.newPassword" type="password" show-password />
                </el-form-item>

                <el-form-item label="确认密码" prop="confirmPassword">
                    <el-input v-model="passwordForm.confirmPassword" type="password" show-password />
                </el-form-item>

                <el-form-item>
                    <el-button type="primary" @click="changePassword">
                        修改密码
                    </el-button>
                </el-form-item>
            </el-form>
        </el-card>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'

const isEditing = ref(false)
const profileFormRef = ref(null)
const passwordFormRef = ref(null)

const profileForm = ref({
    id: '',
    name: '',
    department: '',
    email: '',
    phone: ''
})

const passwordForm = ref({
    oldPassword: '',
    newPassword: '',
    confirmPassword: ''
})

const rules = {
    name: [
        { required: true, message: '请输入姓名', trigger: 'blur' },
        { min: 2, max: 20, message: '长度在 2 到 20 个字符', trigger: 'blur' }
    ],
    email: [
        { required: true, message: '请输入邮箱地址', trigger: 'blur' },
        { type: 'email', message: '请输入正确的邮箱地址', trigger: 'blur' }
    ],
    phone: [
        { required: true, message: '请输入手机号码', trigger: 'blur' },
        { pattern: /^1[3-9]\d{9}$/, message: '请输入正确的手机号码', trigger: 'blur' }
    ]
}

const passwordRules = {
    oldPassword: [
        { required: true, message: '请输入原密码', trigger: 'blur' }
    ],
    newPassword: [
        { required: true, message: '请输入新密码', trigger: 'blur' },
        {
            pattern: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/,
            message: '密码必须包含大小写字母、数字和特殊字符，且长度不少于8位',
            trigger: 'blur'
        }
    ],
    confirmPassword: [
        { required: true, message: '请再次输入新密码', trigger: 'blur' },
        {
            validator: (rule, value, callback) => {
                if (value === '') {
                    callback(new Error('请再次输入密码'))
                } else if (value !== passwordForm.value.newPassword) {
                    callback(new Error('两次输入密码不一致!'))
                } else {
                    callback()
                }
            },
            trigger: 'blur'
        }
    ]
}

// 获取个人信息
const getProfile = async () => {
    try {
        const storedInfo = localStorage.getItem('doctorInfo')
        if (storedInfo) {
            const doctorInfo = JSON.parse(storedInfo)
            profileForm.value = { ...doctorInfo }
        }
    } catch (error) {
        ElMessage.error('获取个人信息失败')
    }
}

// 编辑信息
const handleEdit = () => {
    isEditing.value = true
}

// 取消编辑
const cancelEdit = () => {
    isEditing.value = false
    getProfile()
}

// 提交表单
const submitForm = async () => {
    if (!profileFormRef.value) return

    await profileFormRef.value.validate(async (valid) => {
        if (valid) {
            try {
                // TODO: 调用后端API更新个人信息
                // await axios.put('/api/doctor/profile', profileForm.value)
                ElMessage.success('保存成功')
                isEditing.value = false
                // 更新本地存储的信息
                localStorage.setItem('doctorInfo', JSON.stringify(profileForm.value))
            } catch (error) {
                ElMessage.error('保存失败')
            }
        }
    })
}

// 修改密码
const changePassword = async () => {
    if (!passwordFormRef.value) return

    await passwordFormRef.value.validate(async (valid) => {
        if (valid) {
            try {
                // TODO: 调用后端API修改密码
                // await axios.put('/api/doctor/password', passwordForm.value)
                ElMessage.success('密码修改成功')
                passwordFormRef.value.resetFields()
            } catch (error) {
                ElMessage.error('密码修改失败')
            }
        }
    })
}

onMounted(() => {
    getProfile()
})
</script>

<style scoped>
.doctor-profile {
    padding: 20px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

@media (max-width: 768px) {
    .doctor-profile {
        grid-template-columns: 1fr;
    }
}
</style>