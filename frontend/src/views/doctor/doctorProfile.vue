<!-- 
  doctorProfile.vue
  医生个人信息页面
  功能：展示和编辑医生个人信息，修改密码
  作者：同济大学软件工程管理与经济学项目组
  日期：2025
-->

<template>
  <div class="doctor-profile">
    <!-- 个人信息卡片 -->
    <el-card>
      <template #header>
        <div class="card-header">
          <span>个人信息</span>
          <el-button type="primary" @click="isEditing = true">
            编辑信息
          </el-button>
        </div>
      </template>

      <el-form
        ref="profileFormRef"
        :model="profileForm"
        :rules="formRules.profile"
        label-width="100px"
        :disabled="!isEditing"
      >
        <!-- 基本信息表单项 -->
        <el-form-item
          v-for="field in profileFields"
          :key="field.prop"
          :label="field.label"
          :prop="field.prop"
        >
          <el-input
            v-model="profileForm[field.prop]"
            :disabled="field.alwaysDisabled"
          />
        </el-form-item>

        <!-- 编辑模式下的操作按钮 -->
        <el-form-item v-if="isEditing">
          <el-button
            type="primary"
            @click="submitForm(profileFormRef, 'profile')"
          >
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

      <el-form
        ref="passwordFormRef"
        :model="passwordForm"
        :rules="formRules.password"
        label-width="100px"
      >
        <!-- 密码表单项 -->
        <el-form-item
          v-for="field in passwordFields"
          :key="field.prop"
          :label="field.label"
          :prop="field.prop"
        >
          <el-input
            v-model="passwordForm[field.prop]"
            type="password"
            show-password
          />
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            @click="submitForm(passwordFormRef, 'password')"
          >
            修改密码
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { ElMessage } from "element-plus";

// 表单引用
const profileFormRef = ref(null);
const passwordFormRef = ref(null);

// 编辑状态
const isEditing = ref(false);

// 表单数据
const profileForm = ref({
  id: "",
  name: "",
  department: "",
  email: "",
});

const passwordForm = ref({
  oldPassword: "",
  newPassword: "",
  confirmPassword: "",
});

// 个人信息字段配置
const profileFields = [
  { prop: "id", label: "医生ID", alwaysDisabled: true },
  { prop: "name", label: "姓名", alwaysDisabled: false },
  { prop: "department", label: "科室", alwaysDisabled: true },
  { prop: "email", label: "邮箱", alwaysDisabled: true },
];

// 密码字段配置
const passwordFields = [
  { prop: "oldPassword", label: "原密码" },
  { prop: "newPassword", label: "新密码" },
  { prop: "confirmPassword", label: "确认密码" },
];

// 表单验证规则
const formRules = {
  profile: {
    name: [
      { required: true, message: "请输入姓名", trigger: "blur" },
      { min: 2, max: 20, message: "长度在 2 到 20 个字符", trigger: "blur" },
    ],
  },
  password: {
    oldPassword: [{ required: true, message: "请输入原密码", trigger: "blur" }],
    newPassword: [
      { required: true, message: "请输入新密码", trigger: "blur" },
      {
        pattern:
          /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/,
        message: "密码必须包含大小写字母、数字和特殊字符，且长度不少于8位",
        trigger: "blur",
      },
    ],
    confirmPassword: [
      { required: true, message: "请再次输入新密码", trigger: "blur" },
      {
        validator: (rule, value, callback) => {
          if (!value) {
            callback(new Error("请再次输入密码"));
          } else if (value !== passwordForm.value.newPassword) {
            callback(new Error("两次输入密码不一致!"));
          } else {
            callback();
          }
        },
        trigger: "blur",
      },
    ],
  },
};

/**
 * 获取医生个人信息
 * 从localStorage中获取已存储的医生信息
 */
const getProfile = () => {
  try {
    const storedInfo = localStorage.getItem("doctorInfo");
    if (storedInfo) {
      profileForm.value = { ...JSON.parse(storedInfo) };
    }
  } catch (error) {
    ElMessage.error("获取个人信息失败");
  }
};

/**
 * 取消编辑
 * 重置表单并重新获取个人信息
 */
const cancelEdit = () => {
  isEditing.value = false;
  getProfile();
};

/**
 * 提交表单
 * @param {Ref} formRef - 表单引用
 * @param {string} formType - 表单类型：'profile' 或 'password'
 */
const submitForm = async (formRef, formType) => {
  if (!formRef) return;

  try {
    await formRef.validate();

    if (formType === "profile") {
      // TODO: 调用后端API更新个人信息
      // await updateProfile(profileForm.value)
      localStorage.setItem("doctorInfo", JSON.stringify(profileForm.value));
      isEditing.value = false;
      ElMessage.success("保存成功");
    } else {
      // TODO: 调用后端API修改密码
      // await changePassword(passwordForm.value)
      formRef.resetFields();
      ElMessage.success("密码修改成功");
    }
  } catch (error) {
    ElMessage.error(formType === "profile" ? "保存失败" : "密码修改失败");
  }
};

// 页面加载时获取个人信息
onMounted(getProfile);
</script>

<style scoped>
/* 布局样式 */
.doctor-profile {
  padding: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

/* 卡片头部样式 */
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* 响应式布局 */
@media (max-width: 768px) {
  .doctor-profile {
    grid-template-columns: 1fr;
  }
}
</style>