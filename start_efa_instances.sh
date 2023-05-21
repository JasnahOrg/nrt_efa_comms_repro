#!/bin/bash

# Usage Examples:
# ./start_efa_instances.sh --trn1 --n 2
# ./start_efa_instances.sh --a100 --n 2
# You may need to set some of the variables like REGION or SUBNET or SECURITY_GROUP below

set -e

# AWS CLI v2 Installation instructions for Linux:
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# sudo ./aws/install
# $ aws --version
# aws-cli/2.11.20 Python/3.11.3 Linux/5.15.0-1034-aws exe/x86_64.ubuntu.20 prompt/off
# Someone with AWS console admin privileges can create an access key ID and secret for this:
# Configure credentials: aws configure

# SET THESE!!
SUBNET=subnet-08c4e39e1a0efd37e
SECURITY_GROUP=sg-07e7954d56041233f
KEYNAME=trn

REGION=us-west-2
COUNT=0

# Parse command line arguments
while (( "$#" )); do
  case "$1" in
    --n)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        COUNT=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
      ;;
  esac
done

# Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) 20230517
AMI=ami-01257e71ecb2f431c
INSTANCE_NAME=_Trainium-Big
NETWORK_INTERFACES="NetworkCardIndex=0,DeviceIndex=0,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa \
NetworkCardIndex=1,DeviceIndex=1,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa \
NetworkCardIndex=2,DeviceIndex=1,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa \
NetworkCardIndex=3,DeviceIndex=1,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa \
NetworkCardIndex=4,DeviceIndex=1,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa \
NetworkCardIndex=5,DeviceIndex=1,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa \
NetworkCardIndex=6,DeviceIndex=1,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa \
NetworkCardIndex=7,DeviceIndex=1,Groups=$SECURITY_GROUP,SubnetId=$SUBNET,InterfaceType=efa"

command="aws ec2 --region $REGION run-instances \
--tag-specifications \"ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]\" \
--count $COUNT \
--image-id $AMI \
--instance-type "trn1.32xlarge" \
--key-name $KEYNAME \
--network-interfaces $NETWORK_INTERFACES"

# Conditional part for "a100"
if [ "$TYPE" = "p4d.24xlarge" ]; then
    command+=" --capacity-reservation-specification \"CapacityReservationTarget={CapacityReservationId=cr-0c4df77abe7bc7a17}\""
fi

# Run command
output=$(eval $command)

# Parse the output to get the instance IDs
instance_ids=$(echo $output | jq -r .Instances[].InstanceId)
echo "Got created instance IDs: $instance_ids"

# Loop through each instance ID
public_ips=""
private_ips=""
for instance_id in $instance_ids; do
  echo "Waiting for instance $instance_id to be running..."
  aws ec2 wait instance-running --instance-ids $instance_id --region $REGION

  echo "Creating SSH public IP newtork inteface for instance $instance_id..."
  interface_id=""
  INSTANCE_INFO=$(aws ec2 describe-instances --region $REGION --instance-ids $instance_id)
  OUTPUT=$(echo "$INSTANCE_INFO" | jq -r '.Reservations[0].Instances[0].NetworkInterfaces[] | "\(.Attachment.DeviceIndex),\(.NetworkInterfaceId)"')
  echo $OUTPUT
  for pair in $OUTPUT; do
      IFS="," read -r device_idx ni_id <<< $pair
      if [ "$device_idx" == "0" ]; then
          interface_id=$ni_id
          break
      fi
  done
  if [ "$interface_id" == "" ]; then
      exit -1
  fi
  echo $interface_id

  echo "Checking for unassociated Elastic IPs..."
  unassociated_eips=$(aws ec2 describe-addresses --region $REGION | jq -r '.Addresses[] | select(.AssociationId == null) | .AllocationId')
  if [[ -z "$unassociated_eips" ]]; then
      echo "No unassociated Elastic IPs found. Allocating new Elastic IP..."
      eip_output=$(aws ec2 allocate-address --domain vpc --region $REGION)
      eip_id=$(echo $eip_output | jq -r .AllocationId)
      echo "Allocated Elastic IP ID: $eip_id"
      eip_public_ip=$(echo $eip_output | jq -r .PublicIp)
      echo "Allocated Elastic IP Public IP: $eip_public_ip"
  else
      # use the first unassociated Elastic IP found
      eip_id=$(echo "$unassociated_eips" | head -n 1)
      echo "Found unassociated Elastic IP ID: $eip_id"
      eip_public_ip=$(aws ec2 describe-addresses --allocation-ids $eip_id --region $REGION | jq -r .Addresses[0].PublicIp)
      echo "Elastic IP Public IP: $eip_public_ip"
  fi
  public_ips+="${eip_public_ip} "
  
  echo "Associating Elastic IP with network interface $interface_id..."
  aws ec2 associate-address --allocation-id $eip_id --network-interface-id $interface_id --region $REGION
  echo "Associated Elastic IP with network interface."

  echo "Getting the private IP of network interface $interface_id..."
  interface_info=$(aws ec2 describe-network-interfaces --network-interface-ids $interface_id --region $REGION)
  private_ip=$(echo $interface_info | jq -r '.NetworkInterfaces[0].PrivateIpAddress')
  echo "Private IP of network interface: $private_ip"
  private_ips+="${private_ip} "
done

echo "You can now 'ssh -i $KEYNAME.pem ubuntu@' to $public_ips and check that there are 8 EFA devices with 'lspci -tv' and 'fi_info -p efa -t FI_EP_RDM'"
echo "Private IPs to pass to neuron comms commands: $private_ips"
