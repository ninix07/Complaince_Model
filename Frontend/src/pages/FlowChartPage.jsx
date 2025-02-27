/** @format */

import axios from "axios";
import { useEffect, useState } from "react";
import { ReactFlow } from "reactflow";

import { MiniMap, Controls } from "reactflow";
import "reactflow/dist/style.css";
const createNodesAndEdges = (data) => {
  const nodes = [];
  const edges = [];
  let nodeId = 0;

  // Check if the data is valid
  if (data && typeof data === "object") {
    Object.keys(data).forEach((category) => {
      // Create category node
      const categoryNode = {
        id: `node-${nodeId++}`,
        data: { label: category },
        position: { x: 0, y: nodeId * 100 },
      };
      nodes.push(categoryNode);
      const categoryNodeId = categoryNode.id;

      // Create subcategory nodes and edges
      if (data[category] && typeof data[category] === "object") {
        Object.keys(data[category]).forEach((subcategory) => {
          const subcategoryNode = {
            id: `node-${nodeId++}`,
            data: { label: subcategory },
            position: { x: 200, y: nodeId * 100 },
          };
          nodes.push(subcategoryNode);
          const subcategoryNodeId = subcategoryNode.id;

          // Add edge from category to subcategory
          edges.push({
            id: `edge-${category}-${subcategory}`,
            source: categoryNodeId,
            target: subcategoryNodeId,
          });

          // Create subcategory's data nodes (e.g., 11.1, 4.1)
          if (Array.isArray(data[category][subcategory])) {
            data[category][subcategory].forEach((item) => {
              const itemNode = {
                id: `node-${nodeId++}`,
                data: { label: item },
                position: { x: 400, y: nodeId * 100 },
              };
              nodes.push(itemNode);
              const itemNodeId = itemNode.id;

              // Add edge from subcategory to item
              edges.push({
                id: `edge-${subcategory}-${item}`,
                source: subcategoryNodeId,
                target: itemNodeId,
              });
            });
          }
        });
      }
    });
  }

  return { nodes, edges };
};

const FlowChart = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get("http://127.0.0.1:8080/api/getTaxonomy")
      .then((response) => {
        setData(response.data);
        console.log(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  // Ensure data exists before proceeding
  if (!data) {
    return (
      <div className="flex justify-center items-center h-[70vh] text-2xl font-bold text-red-600">
        Please Upload the CSV file to create the taxonomy first.
      </div>

    );
  }

  const { nodes, edges } = createNodesAndEdges(data);

  return (
    <div style={{ height: "100vh" }}>
      <ReactFlow nodes={nodes} edges={edges} fitView>
        <MiniMap />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default FlowChart;
