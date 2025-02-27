/**
 * @format
 * @type {import('tailwindcss').Config}
 */

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],

  theme: {
    extend: {
      animation: {
        "circle-rotate": "circle-rotate 4s linear infinite",
        "spin-right": "spin-right 2s linear infinite",
        "spin-left": "spin-left 2s linear infinite",
      },

      keyframes: {
        "circle-rotate": {
          "0%": {
            borderColor: "transparent",
          },
          "50%": {
            borderColor: "#000000",
            boxShadow:
              "0 0 0.3rem #fff, 0 0 0.3rem #fff, 0 0 3rem #bc13fe, 0 0 0.9rem #bc13fe, 0 0 2.9rem #bc13fe, inset 0 0 1.4rem #bc13fe",
          },
          "100%": {
            borderColor: "transparent",
          },
        },
        "spin-right": {
          "100%": {
            transform: "rotate(360deg)",
          },
        },
        "spin-left": {
          "100%": {
            transform: "rotate(-360deg)",
          },
        },
      },
      screens: {
        xs: "450px",
        sm: "640px",
        md: "768px",
        lg: "1024px",
      },
      colors: {
        background: {
          light: "#F0F3FE",
          dark: "#12212B",
        },
        text: {
          dark: "#FFFFFF",
          light: "#000000",
        },
        primary: "#303139",
        // secondary: "#CCCCD5",
        secondarys: {
          light: "#000000",
          dark: "#ffffff",
        },
        secondary: "#ffffff",
        tertiary: "#1b2432",
        tertiarylight: "#ffffff",
        "black-100": "#03191E",
        "black-200": "#525174",
        "white-100": "#f3f3f3",
      },
      boxShadow: {
        card: "20px 20px 20px rgba(0, 0, 0, 0.35)",
        "card-dark": "10px 10px 20px rgba(202, 202, 202, 0.25)",
        "card-two": "0px 35px 120px -15px #211e35",
      },
      fontSize: {
        base: "1rem",
        h1: "2.125rem",
        h2: "1.875rem",
        h3: "1.5rem",
      },
      lineHeight: {
        11: "2.75rem",
        12: "3rem",
        13: "3.25rem",
        14: "3.5rem",
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            a: {
              color: theme("colors.primary.500"),
              "&:hover": {
                color: `${theme("colors.primary.600")}`,
              },
              code: { color: theme("colors.primary.400") },
            },
            "h1,h2": {
              fontWeight: "700",
              letterSpacing: theme("letterSpacing.tight"),
            },
            h3: {
              fontWeight: "600",
            },
            code: {
              color: theme("colors.indigo.500"),
            },
          },
        },
        invert: {
          css: {
            a: {
              color: theme("colors.primary.500"),
              "&:hover": {
                color: `${theme("colors.primary.400")}`,
              },
              code: { color: theme("colors.primary.400") },
            },
            "h1,h2,h3,h4,h5,h6": {
              color: theme("colors.gray.100"),
            },
          },
        },
      }),
    },
  },

  variants: {
    extend: {
      boxShadow: ["dark"], // Enable dark mode variants for box shadow
    },
  },

  plugins: [],
};
