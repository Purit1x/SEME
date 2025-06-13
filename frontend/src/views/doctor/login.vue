<!-- 
  login.vue
  医生登录注册页面
  功能：提供医生登录和注册功能，包含表单验证和邮箱验证
  作者：同济大学软件工程管理与经济学项目组
  日期：2025
-->

<template>
  <div class="container" id="app">
    <div class="forms-container">
      <div class="signin-signup">
        <!-- 登录表单 -->
        <el-form
          ref="loginRef"
          :model="loginForm"
          :rules="formRules.login"
          class="sign-in-form"
        >
          <h2 class="title">登录</h2>
          <div
            v-for="field in loginFields"
            :key="field.prop"
            class="input-field"
          >
            <i :class="field.icon"></i>
            <el-form-item :prop="field.prop">
              <el-input
                v-model="loginForm[field.prop]"
                :type="field.type || 'text'"
                :placeholder="field.placeholder"
                :show-password="field.type === 'password'"
                @keyup.enter="handleLogin"
              />
            </el-form-item>
          </div>
          <el-button
            type="primary"
            :loading="loading.login"
            @click="handleLogin"
            class="btn form"
            round
          >
            {{ loading.login ? "登录中" : "登录" }}
          </el-button>
        </el-form>

        <!-- 注册表单 -->
        <el-form
          ref="signUpRef"
          :model="signUpForm"
          :rules="formRules.signup"
          class="sign-up-form"
        >
          <h2 class="title">注册</h2>
          <div
            v-for="field in signupFields"
            :key="field.prop"
            class="input-field"
          >
            <i :class="field.icon"></i>
            <el-form-item :prop="field.prop">
              <template v-if="field.type === 'select'">
                <el-select
                  v-model="signUpForm[field.prop]"
                  :placeholder="field.placeholder"
                  style="width: 100%"
                >
                  <el-option
                    v-for="opt in field.options"
                    :key="opt.value"
                    :label="opt.label"
                    :value="opt.value"
                  />
                </el-select>
              </template>
              <template v-else-if="field.prop === 'verificationCode'">
                <el-input
                  v-model="signUpForm[field.prop]"
                  :placeholder="field.placeholder"
                >
                  <template #append>
                    <el-button
                      type="primary"
                      @click="handleSendCode"
                      :loading="loading.sendCode"
                    >
                      {{ loading.sendCode ? "发送中" : "获取验证码" }}
                    </el-button>
                  </template>
                </el-input>
              </template>
              <template v-else>
                <el-input
                  v-model="signUpForm[field.prop]"
                  :type="field.type || 'text'"
                  :placeholder="field.placeholder"
                  :show-password="field.type === 'password'"
                />
              </template>
            </el-form-item>
          </div>
          <el-button
            type="primary"
            :loading="loading.signup"
            @click="handleSignup"
            class="btn form"
            round
          >
            {{ loading.signup ? "注册中" : "注册" }}
          </el-button>
        </el-form>
      </div>
    </div>

    <!-- 切换面板 -->
    <div class="panels-container">
      <div class="panel left-panel">
        <div class="content">
          <h3>新用户 ?</h3>
          <p>输入您的信息成为我们的客户</p>
          <button class="btn transparent" id="sign-up-btn">注册</button>
        </div>
      </div>
      <div class="panel right-panel">
        <div class="content">
          <h3>已有账号 ?</h3>
          <p>请登录以享受我们更多的服务</p>
          <button class="btn transparent" id="sign-in-btn">登录</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import axios from "axios";
import { ElMessage } from "element-plus";

// 路由实例
const router = useRouter();

// 表单引用
const loginRef = ref(null);
const signUpRef = ref(null);

// 加载状态
const loading = ref({
  login: false,
  signup: false,
  sendCode: false,
});

// 表单数据
const loginForm = ref({
  mail: "",
  password: "",
});

const signUpForm = ref({
  doctor_id: "",
  name: "",
  email: "",
  department: "",
  password: "",
  confirmPassword: "",
  verificationCode: "",
});

// 登录表单字段配置
const loginFields = [
  { prop: "mail", icon: "fa-solid fa-user", placeholder: "邮箱" },
  {
    prop: "password",
    icon: "fa-solid fa-lock",
    type: "password",
    placeholder: "密码",
  },
];

// 注册表单字段配置
const signupFields = [
  { prop: "doctor_id", icon: "fa-solid fa-id-card", placeholder: "医生ID" },
  { prop: "name", icon: "fa-solid fa-user", placeholder: "姓名" },
  { prop: "email", icon: "fa-solid fa-envelope", placeholder: "邮箱" },
  {
    prop: "department",
    icon: "fa-solid fa-hospital",
    type: "select",
    placeholder: "选择科室",
    options: [
      { label: "放射科", value: "放射科" },
      { label: "泌尿外科", value: "泌尿外科" },
    ],
  },
  {
    prop: "password",
    icon: "fa-solid fa-lock",
    type: "password",
    placeholder: "密码",
  },
  {
    prop: "confirmPassword",
    icon: "fa-solid fa-lock",
    type: "password",
    placeholder: "确认密码",
  },
  { prop: "verificationCode", icon: "fa-solid fa-key", placeholder: "验证码" },
];

// 表单验证规则
const formRules = {
  login: {
    mail: [
      { required: true, message: "请输入邮箱", trigger: "blur" },
      { type: "email", message: "请输入正确的邮箱格式", trigger: "blur" },
    ],
    password: [{ required: true, message: "请输入密码", trigger: "blur" }],
  },
  signup: {
    doctor_id: [
      { required: true, message: "请输入医生ID", trigger: "blur" },
      {
        pattern: /^DR\d{3,}$/,
        message: "ID格式为DR开头加3位以上数字",
        trigger: "blur",
      },
    ],
    name: [
      { required: true, message: "请输入姓名", trigger: "blur" },
      { min: 2, max: 20, message: "姓名长度在2-20个字符之间", trigger: "blur" },
    ],
    email: [
      { required: true, message: "请输入邮箱", trigger: "blur" },
      { type: "email", message: "请输入正确的邮箱格式", trigger: "blur" },
    ],
    department: [{ required: true, message: "请选择科室", trigger: "change" }],
    password: [
      { required: true, message: "请输入密码", trigger: "blur" },
      {
        pattern:
          /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/,
        message: "密码必须包含大小写字母、数字和特殊字符，且长度不少于8位",
        trigger: "blur",
      },
    ],
    confirmPassword: [
      { required: true, message: "请确认密码", trigger: "blur" },
      {
        validator: (rule, value, callback) => {
          if (!value) {
            callback(new Error("请再次输入密码"));
          } else if (value !== signUpForm.value.password) {
            callback(new Error("两次输入密码不一致!"));
          } else {
            callback();
          }
        },
        trigger: "blur",
      },
    ],
    verificationCode: [
      { required: true, message: "请输入验证码", trigger: "blur" },
      { len: 6, message: "验证码长度应为6位", trigger: "blur" },
    ],
  },
};

/**
 * 处理登录请求
 */
const handleLogin = async () => {
  if (!loginRef.value) return;

  try {
    await loginRef.value.validate();
    loading.value.login = true;

    const response = await axios.post(
      "http://localhost:5000/api/auth/doctor/login",
      {
        login_id: loginForm.value.mail,
        password: loginForm.value.password,
      }
    );

    if (response.data.success) {
      const { access_token, refresh_token, doctor } = response.data;

      // 存储登录信息
      localStorage.setItem("access_token", access_token);
      localStorage.setItem("refresh_token", refresh_token);
      localStorage.setItem("doctorInfo", JSON.stringify(doctor));

      ElMessage.success("登录成功");
      router.replace("/doctorHome");
    }
  } catch (error) {
    console.error("登录失败:", error);
    ElMessage.error(
      error.response?.data?.message || "登录失败，请检查邮箱和密码"
    );
    loginForm.value.password = "";
  } finally {
    loading.value.login = false;
  }
};

/**
 * 发送验证码
 */
const handleSendCode = async () => {
  // 检查必填字段
  const requiredFields = [
    "doctor_id",
    "name",
    "email",
    "department",
    "password",
  ];
  if (requiredFields.some((field) => !signUpForm.value[field])) {
    ElMessage.warning("请先填写完整注册信息");
    return;
  }

  try {
    loading.value.sendCode = true;
    const response = await axios.post(
      "http://localhost:5000/api/auth/doctor/register",
      {
        doctor_id: signUpForm.value.doctor_id,
        name: signUpForm.value.name,
        password: signUpForm.value.password,
        email: signUpForm.value.email,
        department: signUpForm.value.department,
      }
    );

    if (response.data.success) {
      localStorage.setItem("verification_id", response.data.verification_id);
      ElMessage.success("验证码已发送，请查收邮件");
    }
  } catch (error) {
    ElMessage.error(
      "验证码发送失败：" + error.response?.data?.message || error.message
    );
  } finally {
    loading.value.sendCode = false;
  }
};

/**
 * 处理注册请求
 */
const handleSignup = async () => {
  const verification_id = localStorage.getItem("verification_id");
  if (!verification_id) {
    ElMessage.error("请先获取验证码");
    return;
  }

  try {
    await signUpRef.value.validate();
    loading.value.signup = true;

    const response = await axios.post(
      "http://localhost:5000/api/auth/doctor/verify",
      {
        verification_id,
        code: signUpForm.value.verificationCode,
      }
    );

    if (response.data.success) {
      const { access_token, refresh_token } = response.data;

      // 存储token
      localStorage.setItem("access_token", access_token);
      localStorage.setItem("refresh_token", refresh_token);

      ElMessage.success("注册成功");
      signUpRef.value.resetFields();
      localStorage.removeItem("verification_id");

      // 切换到登录界面
      document.querySelector("#sign-in-btn").click();
    }
  } catch (error) {
    ElMessage.error(
      "验证失败：" + error.response?.data?.message || error.message
    );
  } finally {
    loading.value.signup = false;
  }
};

// 页面加载时的处理
onMounted(() => {
  // 检查是否已登录
  const data = localStorage.getItem("microData");
  if (data) {
    router.push("/patientHome");
    return;
  }

  // 设置表单切换动画
  const container = document.querySelector(".container");
  document.querySelector("#sign-in-btn").addEventListener("click", () => {
    container.classList.remove("sign-up-mode");
  });
  document.querySelector("#sign-up-btn").addEventListener("click", () => {
    container.classList.add("sign-up-mode");
  });
});
</script>

<style>
@import "../../assets/doctor/css/style.css";

/* Element Plus 样式覆盖 */
.el-button.is-round {
  border-radius: 49px;
}

.el-form-item {
  margin-bottom: 0;
}

.el-form-item.is-error .el-input__inner,
.el-form-item.is-error .el-input__inner:focus,
.el-form-item.is-error .el-textarea__inner,
.el-form-item.is-error .el-textarea__inner:focus {
  box-shadow: none;
}

/* 输入框样式 */
.el-input {
  align-items: center;
  margin-right: 8px;
}

.el-input .el-input__icon {
  font-size: 1.4em;
}

/* 下拉选择框样式 */
.el-select .el-input {
  width: 100%;
}

.input-field .el-select {
  width: 100%;
}

/* 验证码按钮样式 */
.el-input-group__append {
  padding: 0;
}

.el-input-group__append button {
  border: none;
  height: 100%;
  border-radius: 0;
}
</style>