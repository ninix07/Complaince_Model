/** @format */

import axios from "axios";
import { useState, useEffect } from "react";
import Chart from "react-apexcharts";
import LoadingAnimation from "../components/Loading.jsx";

const ComplianceCharts = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get("http://127.0.0.1:8080/api/countCompliance")
      .then((response) => {
        setData(response.data);
        setLoading(false);
        console.log(data)
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingAnimation />;

  const categories = data.map((item) => item.compliance);
  const values = data.map((item) => item.total_questions);
  const minValue = Math.min(...values);
  const minCompliance = categories[values.indexOf(minValue)];

  const getRandomColor = () => {
    const colors = [
      "#FFD700",
      "#ADD8E6",
      "#90EE90",
      "#FFB6C1",
      "#FFE4B5",
      "#D8BFD8",
      "#E6E6FA",
      "#F0E68C",
      "#B0E0E6",
      "#FAFAD2",
      "#F5DEB3",
      "#E0FFFF",
      "#FFFACD",
      "#FFDEAD",
      "#C1FFC1"
    ];

    return colors[Math.floor(Math.random() * colors.length)];
  };
  const colors = values.map((value) =>
    value === minValue ? "#FF0000" : getRandomColor()
  );

  const barOptions = {
    chart: { type: "bar" },
    toolbar: {
      style: {
        background: "#1E1E1E",
      },
    },
    tooltip: {
      theme: "dark",
      style: {
        background: "#1E1E1E",
        color: "#FFFFFF",
      },
    },

    xaxis: {
      categories,
      labels: {
        style: {
          colors: "#FFFFFF",
        },
      },
    },
    plotOptions: {
      bar: { distributed: true },
    },
    legend: {
      labels: {
        colors: "#FFFFFF",
      },
    },
    colors,
    dataLabels: {
      enabled: true,
      style: {
        fontSize: "14px",
        fontWeight: "bold",
        colors: ["#FFFFFF"],
      },
    },
  };

  const pieOptions = {
    labels: categories,
    colors,
    legend: {
      labels: {
        colors: "#FFFFFF",
      },
    },
    toolbar: {
      style: {
        background: "#1E1E1E",
      },
    },
    tooltip: {
      theme: "dark",
      style: {
        background: "#1E1E1E",
        color: "#FFFFFF",
      },
    },
    dataLabels: {
      enabled: true,
      style: {
        fontSize: "14px",
        fontWeight: "bold",
        colors: ["#FFFFFF"],
      },
    },
  };
  if (!data) {
    return (
      <div className="flex justify-center items-center h-[70vh] text-2xl font-bold text-red-600">
        Trouble Connecting to Backend.
      </div>

    );
  }
  return (
    <div className="flex flex-col  items-center w-[80vw] min-h-screen p-6">
      <div className="flex flex-col items-center w-[100vw] space-y-6">

        {/* Title */}
        <h1 className="text-4xl font-bold text-center text-white-800">
          Compliance Dashboard
        </h1>

        {/* Charts Section */}
        <div className="flex flex-wrap justify-center items-center gap-8 w-full">
          <Chart
            options={barOptions}
            series={[{ data: values }]}
            type="bar"
            height={350}
            width={400}
          />
          <Chart
            options={pieOptions}
            series={values}
            type="pie"
            height={350}
            width={400}
          />
        </div>

        {/* Compliance Message */}
        <p className="text-lg text-white-700 text-center max-w-[700px]">
          The compliance <strong className="text-blue-600">{minCompliance}</strong> has the lowest number of
          questions compared to others. Asking more questions from it might be
          beneficial.
        </p>

      </div>
    </div>
  );
};

export default ComplianceCharts;
