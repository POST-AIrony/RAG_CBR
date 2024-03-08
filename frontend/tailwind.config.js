/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./widgets/**/*.{js,ts,jsx,tsx,mdx}",
    "./shared/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      screens: {
        deskWide: { min: "1560px" },
        mlarge: { max: "480px" },
        mmedium: { max: "380px" },
        msmall: { max: "320px" },
      },
    },
  },
  plugins: [],
};
